#!/usr/bin/env python3
"""
Advanced multi-output regression (WIP, Investment, Satisfaction) with:
- Multi-head MLP (shared trunk + one head per target)
- Weighted loss (prioritize satisfaction)
- Output calibration (per-target affine calibration on validation set)
- Sklearn-like API (fit / predict), easy to extend for ensembling/stacking

This file is meant to live in: src/models/advanced_multioutput_torch.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Utilities
# -----------------------------
def _as_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch tensor to numpy (on CPU)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _set_torch_seed(seed: int) -> None:
    """Set torch random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism note: full determinism can slow down; keep default unless needed.


# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class ModelConfig:
    # Model architecture
    input_dim: int
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    activation: str = "relu"  # "relu" or "gelu"
    num_targets: int = 3  # wip, investment, satisfaction

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 50
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda" if available

    # Weighted loss (prioritize satisfaction)
    # Convention: target order is [wip, investment, satisfaction]
    loss_weights: Tuple[float, float, float] = (0.2, 0.2, 0.6)

    # Calibration
    # If targets are known to be in [0, 1], clipping is recommended.
    calibrate: bool = True
    clip_min: Optional[float] = 0.0
    clip_max: Optional[float] = 1.0

    # Early stopping (simple)
    early_stopping: bool = True
    patience: int = 8


# -----------------------------
# Multi-head MLP
# -----------------------------
class MultiHeadMLP(nn.Module):
    """
    Shared trunk + separate heads for each target.
    This is often better than one big linear layer for multi-output regression,
    because each KPI can require different last-layer mappings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        activation: str,
        num_targets: int,
    ):
        super().__init__()

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), act(), nn.Dropout(dropout)]
            dim = h
        self.trunk = nn.Sequential(*layers)

        # One head per target (scalar output each)
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_targets)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape (batch_size, num_targets).
        """
        z = self.trunk(x)
        outs = [head(z) for head in self.heads]  # list of (B,1)
        y = torch.cat(outs, dim=1)  # (B, num_targets)
        return y


# -----------------------------
# Weighted loss
# -----------------------------
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE per target:
    loss = sum_i w_i * mean((y_pred_i - y_true_i)^2)

    Why MSE?
    - Smooth gradients
    - Usually works well as a base loss for regression

    If you want to optimize tolerance-metric more directly, you can later replace
    this with a custom differentiable approximation (e.g., smooth indicator).
    """

    def __init__(self, weights: Tuple[float, ...]):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # (B, T)
        mse_per_target = torch.mean((y_pred - y_true) ** 2, dim=0)  # (T,)
        return torch.sum(self.weights * mse_per_target)


# -----------------------------
# Output calibration (affine)
# -----------------------------
class AffineCalibrator:
    """
    Simple per-target affine calibration:
        y_cal = a * y_pred + b

    Fit (a,b) on validation set by least squares:
        minimize ||(a*y_pred + b) - y_true||^2

    Pros:
    - Very stable
    - Cheap
    - Often improves "within tolerance" metrics when model is biased/under-scaled

    Optional clipping if targets are bounded (e.g., [0,1]).
    """

    def __init__(self, num_targets: int, clip_min: Optional[float], clip_max: Optional[float]):
        self.num_targets = num_targets
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.a = np.ones(num_targets, dtype=np.float32)
        self.b = np.zeros(num_targets, dtype=np.float32)
        self.fitted = False

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "AffineCalibrator":
        """
        y_pred, y_true: shape (N, T)
        """
        assert y_pred.shape == y_true.shape
        T = y_pred.shape[1]
        assert T == self.num_targets

        # Fit per target independently (simple linear regression with intercept).
        for t in range(T):
            x = y_pred[:, t]
            y = y_true[:, t]

            # Solve for a,b in least squares: y ≈ a*x + b
            # Using normal equations for 2-parameter regression.
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
            var_x = float(np.mean((x - x_mean) ** 2))

            # Handle degenerate case (var_x ~ 0)
            if var_x < 1e-12:
                a = 1.0
                b = y_mean - a * x_mean
            else:
                a = cov_xy / var_x
                b = y_mean - a * x_mean

            self.a[t] = a
            self.b[t] = b

        self.fitted = True
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration + optional clipping.
        """
        if not self.fitted:
            return self._clip(y_pred)

        y_cal = y_pred * self.a.reshape(1, -1) + self.b.reshape(1, -1)
        return self._clip(y_cal)

    def _clip(self, y: np.ndarray) -> np.ndarray:
        if self.clip_min is not None:
            y = np.maximum(y, self.clip_min)
        if self.clip_max is not None:
            y = np.minimum(y, self.clip_max)
        return y


# -----------------------------
# Main estimator (sklearn-like)
# -----------------------------
class AdvancedMultiOutputRegressor:
    """
    Sklearn-like wrapper around a torch multi-head MLP.

    Features:
    - fit(X_train, y_train, X_val, y_val)
    - predict(X)
    - output calibration after training using val set
    - easy to extend:
        * swap model (replace MultiHeadMLP)
        * swap loss (replace WeightedMSELoss)
        * add ensembling (see EnsembleRegressor below)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        _set_torch_seed(config.seed)

        self.device = torch.device(
            "cuda" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.model = MultiHeadMLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
            num_targets=config.num_targets,
        ).to(self.device)

        self.criterion = WeightedMSELoss(config.loss_weights).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        self.calibrator = AffineCalibrator(
            num_targets=config.num_targets, clip_min=config.clip_min, clip_max=config.clip_max
        )

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "AdvancedMultiOutputRegressor":
        """
        Train the model. If (X_val, y_val) provided and config.calibrate=True:
        - fit calibrator on validation predictions
        - optionally use early stopping on validation loss
        """
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_val = float("inf")
        best_state = None
        patience_left = self.config.patience

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            # Optional validation loop
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val)

                # Early stopping logic
                if self.config.early_stopping:
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_left = self.config.patience
                    else:
                        patience_left -= 1

            if verbose:
                if val_loss is None:
                    print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}")
                else:
                    print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if self.config.early_stopping and X_val is not None and y_val is not None:
                if patience_left <= 0:
                    if verbose:
                        print(f"Early stopping triggered. Best val_loss={best_val:.6f}")
                    break

        # Restore best model if we used early stopping and captured best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        # Fit calibrator on validation predictions (recommended)
        if self.config.calibrate and X_val is not None and y_val is not None:
            y_val_pred = self.predict_raw(X_val)
            self.calibrator.fit(y_val_pred, y_val)

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Raw model predictions without calibration.
        """
        assert self.is_fitted, "Call fit() before predict()."
        self.model.eval()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(X_t)
        return _as_numpy(pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Final predictions (raw + optional calibration).
        """
        y_pred = self.predict_raw(X)
        return self.calibrator.transform(y_pred)

    def _eval_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute weighted MSE loss on a dataset (no gradient).
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(X_t)
            loss = self.criterion(pred, y_t)
        return float(loss.item())


# -----------------------------
# Optional: simple ensemble wrapper
# -----------------------------
class EnsembleRegressor:
    """
    Minimal ensemble combiner.

    - By default: average predictions of multiple fitted models.
    - You can later replace with stacking (train a meta-model) easily.
    """

    def __init__(self, models: List[AdvancedMultiOutputRegressor], weights: Optional[List[float]] = None):
        assert len(models) > 0
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models), dtype=np.float32) / len(models)
        else:
            w = np.asarray(weights, dtype=np.float32)
            self.weights = w / np.sum(w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]  # list of (N,T)
        stacked = np.stack(preds, axis=0)  # (M,N,T)
        return np.tensordot(self.weights, stacked, axes=(0, 0))  # (N,T)

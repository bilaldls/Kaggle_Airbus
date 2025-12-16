#!/usr/bin/env python3
"""
Preprocessing pipeline for multi-output regression (wip, investment, satisfaction).

This script:
- Loads train.csv and test.csv
- Splits features X and targets y
- Builds a robust sklearn preprocessing pipeline:
    * numeric: imputation + scaling
    * categorical: imputation + one-hot encoding
- Produces:
    X_train_proc, y_train
    X_val_proc, y_val
    X_test_proc
- Returns fitted preprocessor for reuse

This file is meant to live in: src/data/preprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class PreprocessConfig:
    train_path: str = "data/raw/train.csv"
    test_path: str = "data/raw/test.csv"

    # Target columns as defined by the hackathon statement
    target_cols: Tuple[str, str, str] = ("wip", "investment", "satisfaction")

    # If you have an ID column, keep it for submission
    id_col: str = "id"

    # Split
    val_size: float = 0.2
    seed: int = 42

    # Scaling
    scale_numeric: bool = True

    # Output type
    # If targets are in [0,1], you can clip later in the model calibrator.
    # Keep raw y here; do not transform targets unless needed.
    pass


# -----------------------------
# Core functions
# -----------------------------
def load_raw_data(cfg: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs."""
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)
    return train_df, test_df


def infer_feature_columns(train_df: pd.DataFrame, cfg: PreprocessConfig) -> List[str]:
    """
    Determine which columns are features:
    - exclude target columns
    - exclude id column if present (we keep id separately)
    """
    excluded = set(cfg.target_cols)
    if cfg.id_col in train_df.columns:
        excluded.add(cfg.id_col)
    feature_cols = [c for c in train_df.columns if c not in excluded]
    return feature_cols


def build_preprocessor(X: pd.DataFrame, cfg: PreprocessConfig) -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer that handles numeric and categorical columns.

    Numeric pipeline:
    - impute missing values with median
    - optional standard scaling

    Categorical pipeline:
    - impute missing values with most_frequent
    - one-hot encode (handle_unknown='ignore' is mandatory for Kaggle)
    """
    # Identify column types from the dataframe
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Numeric processing
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if cfg.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    # Categorical processing
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def preprocess_fit_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, np.ndarray, np.ndarray
]:
    """
    Main preprocessing entry point.

    Returns:
    - X_train_proc, y_train
    - X_val_proc, y_val
    - X_test_proc
    - fitted preprocessor
    - train_ids, val_ids, test_ids (useful for tracking and submission)
    """
    # Keep IDs if present
    train_ids = train_df[cfg.id_col].to_numpy() if cfg.id_col in train_df.columns else None
    test_ids = test_df[cfg.id_col].to_numpy() if cfg.id_col in test_df.columns else None

    # Extract y and X
    y = train_df[list(cfg.target_cols)].to_numpy(dtype=np.float32)

    feature_cols = infer_feature_columns(train_df, cfg)
    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Split BEFORE fitting the preprocessor (prevents leakage)
    X_train, X_val, y_train, y_val, idx_train, idx_val = _train_val_split_with_indices(
        X, y, cfg, train_ids
    )

    # Fit preprocessor on TRAIN only
    preprocessor = build_preprocessor(X_train, cfg)
    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_proc = preprocessor.transform(X_val).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    # IDs per split (optional)
    if train_ids is not None:
        train_ids_split = train_ids[idx_train]
        val_ids_split = train_ids[idx_val]
    else:
        train_ids_split = None
        val_ids_split = None

    return (
        X_train_proc, y_train.astype(np.float32),
        X_val_proc, y_val.astype(np.float32),
        X_test_proc,
        preprocessor,
        train_ids_split, val_ids_split,
        test_ids
    )


def _train_val_split_with_indices(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: PreprocessConfig,
    train_ids: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper that returns train/val split + indices (for id tracking).
    """
    n = len(X)
    indices = np.arange(n)

    idx_train, idx_val = train_test_split(
        indices,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        shuffle=True,
    )

    X_train = X.iloc[idx_train].copy()
    X_val = X.iloc[idx_val].copy()
    y_train = y[idx_train]
    y_val = y[idx_val]

    return X_train, X_val, y_train, y_val, idx_train, idx_val

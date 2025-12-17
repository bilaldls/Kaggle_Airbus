#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

import lightgbm as lgb
from sklearn.model_selection import KFold


# =========================
# Paths / config
# =========================
DATA = Path("data")
X_PATH = DATA / "X_train_clean.parquet"   # <- IMPORTANT: no id
Y_PATH = DATA / "y_train.parquet"
ID_COL = "id"
TARGET = "satisfaction"
N_SPLITS = 5
SEED = 42

TOL = 0.05  # Kaggle metric: absolute ±0.05

NUM_BOOST_ROUND = 20000
EARLY_STOPPING_ROUNDS = 300


# =========================
# Metric / utils
# =========================
def acc_within_tol(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.05) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def lgb_acc_within_tol(y_pred: np.ndarray, dataset: lgb.Dataset):
    y_true = dataset.get_label()
    return ("acc@0.05", acc_within_tol(y_true, y_pred, TOL), True)


def best_shift_for_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float = 0.05,
    shift_range: float = 0.05,
    steps: int = 401
):
    # grille fine suffisante pour satisfaction (0..1)
    shifts = np.linspace(-shift_range, shift_range, steps)
    best_s, best_score = 0.0, -1.0
    for s in shifts:
        score = acc_within_tol(y_true, y_pred + s, tol)
        if score > best_score:
            best_score = score
            best_s = float(s)
    return best_s, best_score


def clip_pred(pred: np.ndarray, y_train_fold: np.ndarray) -> np.ndarray:
    lo = float(np.nanpercentile(y_train_fold, 0.1))
    hi = float(np.nanpercentile(y_train_fold, 99.9))
    return np.clip(pred, lo, hi)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    print("=== Load X / y ===")
    Xdf = pl.read_parquet(X_PATH)
    ydf = pl.read_parquet(Y_PATH)

    if TARGET not in ydf.columns:
        raise ValueError(f"Missing target '{TARGET}' in {Y_PATH}")

    # Drop id if present
    if ID_COL in Xdf.columns:
        print("⚠️ Found 'id' in X -> dropping it for training.")
        Xdf = Xdf.drop(ID_COL)

    X = Xdf.to_numpy().astype(np.float32, copy=False)
    y = ydf.select(TARGET).to_numpy().ravel().astype(np.float32)

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    return X, y


# =========================
# Params / model runner
# =========================
def base_params(seed: int = 42) -> dict:
    # config stable pour acc@0.05
    return dict(
        learning_rate=0.03,
        num_leaves=255,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=80,
        reg_lambda=1.0,
        reg_alpha=0.0,
        verbosity=-1,
        seed=seed,
    )


def objective_params(objective_name: str) -> dict:
    """
    objective_name in {"l2", "huber"}
    """
    if objective_name == "l2":
        return dict(objective="regression", metric="l2")  # MSE
    if objective_name == "huber":
        # selon versions, "huber" peut ne pas être dispo
        return dict(objective="huber", metric="l1")
    raise ValueError(f"Unknown objective_name: {objective_name}")


def try_train_objective(X: np.ndarray, y: np.ndarray, objective_name: str) -> dict:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(y), dtype=np.float32)
    fold_scores = []
    best_iters = []

    print(f"\n==============================")
    print(f"=== Phase 1 | objective = {objective_name.upper()} ===")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        params = {}
        params.update(base_params(seed=SEED))
        params.update(objective_params(objective_name))

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            num_boost_round=NUM_BOOST_ROUND,
            feval=lgb_acc_within_tol,  # métrique Kaggle
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        pred = model.predict(X_va, num_iteration=model.best_iteration).astype(np.float32)
        pred = clip_pred(pred, y_tr)

        oof[va_idx] = pred
        best_iters.append(model.best_iteration)

        acc_fold = acc_within_tol(y_va, pred, TOL)
        fold_scores.append(acc_fold)

        print(f"[Fold {fold}] best_iter={model.best_iteration} | acc@0.05(raw)={acc_fold:.4f}")

    acc_oof_raw = acc_within_tol(y, oof, TOL)
    shift, acc_oof_shift = best_shift_for_metric(y, oof, TOL, shift_range=0.05, steps=401)
    mean_best_iter = int(np.round(np.mean(best_iters)))

    print("\n--- OOF summary ---")
    print(f"OOF acc@0.05 (raw clipped) : {acc_oof_raw:.4f}")
    print(f"Best global shift          : {shift:+.5f}")
    print(f"OOF acc@0.05 (after shift) : {acc_oof_shift:.4f}")
    print(f"Fold acc mean/std          : {np.mean(fold_scores):.4f} / {np.std(fold_scores):.4f}")
    print(f"Mean best_iter             : {mean_best_iter}")

    return dict(
        objective=objective_name,
        acc_raw=acc_oof_raw,
        acc_shifted=acc_oof_shift,
        shift=shift,
        mean_best_iter=mean_best_iter,
    )


def main():
    X, y = load_data()

    results = []

    # Always run L2
    results.append(try_train_objective(X, y, "l2"))

    # Try huber, but handle if not supported
    try:
        results.append(try_train_objective(X, y, "huber"))
    except Exception as e:
        print("\n⚠️ Huber objective failed on your setup.")
        print("Reason:", repr(e))
        print("=> We'll keep L2 as candidate.\n")

    # pick best
    best = max(results, key=lambda r: r["acc_shifted"])

    print("\n==============================")
    print("=== PHASE 1 RESULT ===")
    for r in results:
        print(
            f"{r['objective']:>6s} | raw={r['acc_raw']:.4f} | shifted={r['acc_shifted']:.4f} "
            f"| shift={r['shift']:+.5f} | mean_iter={r['mean_best_iter']}"
        )

    print("\n✅ Best objective:", best["objective"])
    print(f"✅ Best OOF acc@0.05(after shift): {best['acc_shifted']:.4f}")

    # save chosen config for phase 2
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / "phase1_best_objective.txt").write_text(best["objective"], encoding="utf-8")
    (out / "phase1_best_shift.txt").write_text(f"{best['shift']:.8f}", encoding="utf-8")
    (out / "phase1_mean_best_iter.txt").write_text(str(best["mean_best_iter"]), encoding="utf-8")

    print("\nSaved Phase 1 selection to artifacts/:")
    print(" - phase1_best_objective.txt")
    print(" - phase1_best_shift.txt")
    print(" - phase1_mean_best_iter.txt")


if __name__ == "__main__":
    main()
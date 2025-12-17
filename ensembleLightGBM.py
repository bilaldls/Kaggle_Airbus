#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

import lightgbm as lgb
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")  # pour sauvegarder sans fenêtre (VS Code / terminal)
import matplotlib.pyplot as plt


# =========================
# Paths / config
# =========================
DATA = Path("data")
X_PATH = DATA / "X_train_clean.parquet"  # 5528 features (const+collinear removed)
Y_PATH = DATA / "y_train.parquet"

ID_COL = "id"
TARGET = "satisfaction"

N_SPLITS = 5
KF_SEED = 42

# Ensemble seeds (3–5 souvent utile)
SEEDS = [1, 2, 3, 4, 5]

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

    if ID_COL in Xdf.columns:
        print("⚠️ Found 'id' in X -> dropping it for training.")
        Xdf = Xdf.drop(ID_COL)

    # Ensure numeric only (optional safety)
    # If you know all are numeric you can remove this block.
    numeric_cols = [c for c, dt in zip(Xdf.columns, Xdf.dtypes) if dt in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )]
    Xdf = Xdf.select(numeric_cols)

    X = Xdf.to_numpy().astype(np.float32, copy=False)
    y = ydf.select(TARGET).to_numpy().ravel().astype(np.float32)

    print(f"X shape: {X.shape} | y shape: {y.shape} | #features={X.shape[1]}")
    return X, y

def save_error_analysis(y: np.ndarray, pred: np.ndarray, out_dir: Path, tol: float = 0.05, prefix: str = "lgbm_oof"):
    out_dir.mkdir(exist_ok=True)

    err = (y - pred).astype(np.float32)
    abs_err = np.abs(err)

    # ---- stats text ----
    stats = []
    stats.append(f"acc@{tol:.2f}={np.mean(abs_err <= tol):.6f}")
    stats.append(f"MAE={np.mean(abs_err):.6f}")
    stats.append(f"RMSE={np.sqrt(np.mean(err**2)):.6f}")
    stats.append(f"median_abs_err={np.median(abs_err):.6f}")

    # combien sont "presque bons" (pile utile pour gagner acc@0.05)
    for a, b in [(0.05, 0.06), (0.06, 0.07), (0.07, 0.08), (0.08, 0.10)]:
        pct = float(np.mean((abs_err > a) & (abs_err <= b)) * 100.0)
        stats.append(f"pct_|err|_in_({a:.2f},{b:.2f}]={pct:.3f}%")

    (out_dir / f"{prefix}_error_stats.txt").write_text("\n".join(stats), encoding="utf-8")
    print(f"\n✅ Saved stats: {out_dir / f'{prefix}_error_stats.txt'}")

    # ---- CSV détaillé ----
    df = pl.DataFrame({
        "y_true": y.astype(np.float32),
        "y_pred": pred.astype(np.float32),
        "err": err,
        "abs_err": abs_err,
        "good@tol": (abs_err <= tol),
    })
    csv_path = out_dir / f"{prefix}_errors.csv"
    df.write_csv(csv_path)
    print(f"✅ Saved errors CSV: {csv_path}")

    # ---- plots ----
    # 1) histogram abs_err (avec seuil)
    plt.figure()
    plt.hist(abs_err, bins=200)
    plt.axvline(tol, linestyle="--")
    plt.title(f"|error| distribution (tol={tol})")
    plt.xlabel("|y - pred|")
    plt.ylabel("count")
    p1 = out_dir / f"{prefix}_abs_error_hist.png"
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot: {p1}")

    # 2) histogram err
    plt.figure()
    plt.hist(err, bins=200)
    plt.axvline(0.0, linestyle="--")
    plt.title("error distribution (y - pred)")
    plt.xlabel("error")
    plt.ylabel("count")
    p2 = out_dir / f"{prefix}_error_hist.png"
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot: {p2}")

    # 3) calibration bins: mean(y_true) vs mean(pred)
    bins = np.linspace(0.0, 1.0, 21)
    idx = np.digitize(pred, bins) - 1
    xs, ys, ns = [], [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if m.sum() < 50:
            continue
        xs.append(float(np.mean(pred[m])))
        ys.append(float(np.mean(y[m])))
        ns.append(int(m.sum()))

    if len(xs) >= 2:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("calibration (bin means)")
        plt.xlabel("mean(pred)")
        plt.ylabel("mean(y_true)")
        p3 = out_dir / f"{prefix}_calibration.png"
        plt.savefig(p3, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot: {p3}")

# =========================
# Params / model runner
# =========================
def lgb_params(seed: int) -> dict:
    # Base config (from your good results)
    return dict(
        objective="regression",   # L2 / MSE
        metric="l2",
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


def run_cv_one_seed(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float, list[int]]:
    """
    Returns:
      - oof_pred: (n,) OOF predictions for this seed
      - oof_acc:  OOF acc@0.05 for this seed (raw clipped)
      - best_iters per fold
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KF_SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    best_iters = []

    fold_accs = []

    print(f"\n--- Seed {seed} ---")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        model = lgb.train(
            lgb_params(seed),
            dtr,
            valid_sets=[dva],
            num_boost_round=NUM_BOOST_ROUND,
            feval=lgb_acc_within_tol,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        pred = model.predict(X_va, num_iteration=model.best_iteration).astype(np.float32)
        pred = clip_pred(pred, y_tr)

        oof[va_idx] = pred
        best_iters.append(model.best_iteration)

        acc_fold = acc_within_tol(y_va, pred, TOL)
        fold_accs.append(acc_fold)

        print(f"[Seed {seed} | Fold {fold}] best_iter={model.best_iteration} | acc@0.05={acc_fold:.4f}")

    oof_acc = acc_within_tol(y, oof, TOL)
    print(f"[Seed {seed}] OOF acc@0.05 (raw clipped): {oof_acc:.4f} | fold mean/std: {np.mean(fold_accs):.4f}/{np.std(fold_accs):.4f}")
    return oof, oof_acc, best_iters


def main():
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    X, y = load_data()

    print("\n=== Stage: 5-Fold CV for each seed ===")
    oof_list = []
    seed_scores = []
    all_best_iters = []

    for s in SEEDS:
        oof_s, acc_s, best_iters = run_cv_one_seed(X, y, s)
        oof_list.append(oof_s)
        seed_scores.append(acc_s)
        all_best_iters.extend(best_iters)

    # Ensemble: mean of OOF predictions across seeds
    oof_ens = np.mean(np.vstack(oof_list), axis=0).astype(np.float32)
    acc_ens_raw = acc_within_tol(y, oof_ens, TOL)
    save_error_analysis(y, oof_ens, out, tol=TOL, prefix="lgbm_oof_raw")

    shift, acc_ens_shift = best_shift_for_metric(y, oof_ens, TOL, shift_range=0.05, steps=401)
    save_error_analysis(y, oof_ens + shift, out, tol=TOL, prefix="lgbm_oof_shifted")
    mean_best_iter = int(np.round(np.mean(all_best_iters)))

    print("\n==============================")
    print("=== ENSEMBLE OOF SUMMARY ===")
    print(f"Seeds: {SEEDS}")
    print(f"Seed acc mean/std        : {np.mean(seed_scores):.4f} / {np.std(seed_scores):.4f}")
    print(f"Ensemble OOF acc (raw)   : {acc_ens_raw:.4f}")
    print(f"Best global shift        : {shift:+.5f}")
    print(f"Ensemble OOF acc (shift) : {acc_ens_shift:.4f}")
    print(f"Global mean best_iter    : {mean_best_iter}")

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    # Save report + shift
    (out / "phase2_ensemble_seeds.txt").write_text(
        "\n".join([
            f"seeds={SEEDS}",
            f"seed_acc_mean={np.mean(seed_scores):.6f}",
            f"seed_acc_std={np.std(seed_scores):.6f}",
            f"ensemble_oof_acc_raw={acc_ens_raw:.6f}",
            f"ensemble_oof_shift={shift:.8f}",
            f"ensemble_oof_acc_shift={acc_ens_shift:.6f}",
            f"mean_best_iter={mean_best_iter}",
        ]),
        encoding="utf-8"
    )
    (out / "calibration_shift_satisfaction.txt").write_text(f"{shift:.8f}", encoding="utf-8")

    print("\n✅ Saved:")
    print(" - artifacts/phase2_ensemble_seeds.txt")
    print(" - artifacts/calibration_shift_satisfaction.txt")

    # OPTIONAL: refit final models on full train for each seed (for test prediction later)
    print("\n=== Refit final models on full train (one per seed) ===")
    dfull = lgb.Dataset(X, label=y)

    for s in SEEDS:
        model = lgb.train(
            lgb_params(s),
            dfull,
            num_boost_round=mean_best_iter,  # use average best_iter
            valid_sets=None
        )
        model_path = out / f"lgbm_satisfaction_full_seed{s}.txt"
        model.save_model(str(model_path))
        print(f"Saved: {model_path}")

    print("\n✅ Phase 2 done. Next step: predict test.csv and build submission with shift + ensemble mean.")


if __name__ == "__main__":
    main()
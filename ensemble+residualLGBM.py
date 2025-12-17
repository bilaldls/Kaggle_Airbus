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
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA / "X_train_clean.parquet"
Y_TRAIN_PATH = DATA / "y_train.parquet"
X_TEST_PATH  = DATA / "X_test_clean.parquet"
TEST_RAW_CSV = DATA / "test.csv"   # contient id

ID_COL = "id"
TARGET = "satisfaction"
TOL = 0.05

N_SPLITS = 5
KF_SEED = 42

# Base ensemble
BASE_SEEDS = [1, 2, 3, 4, 5]
BASE_NUM_BOOST = 30000
BASE_EARLY_STOP = 300

# Residual ensemble (plus petit)
RES_SEEDS = [11, 12, 13]
RES_NUM_BOOST = 20000
RES_EARLY_STOP = 300

# Shift search
SHIFT_RANGE = 0.05
SHIFT_STEPS = 401


# =========================
# Utils / metric
# =========================
def acc_within_tol(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.05) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def best_shift_for_metric(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.05,
                          shift_range: float = 0.05, steps: int = 401) -> tuple[float, float]:
    shifts = np.linspace(-shift_range, shift_range, steps)
    best_s, best_score = 0.0, -1.0
    for s in shifts:
        score = acc_within_tol(y_true, y_pred + s, tol)
        if score > best_score:
            best_score = score
            best_s = float(s)
    return best_s, best_score

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _numeric_cols(df: pl.DataFrame) -> list[str]:
    ok = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
          pl.Float32, pl.Float64)
    return [c for c, dt in zip(df.columns, df.dtypes) if dt in ok]


# =========================
# Data loading
# =========================
def load_train() -> tuple[np.ndarray, np.ndarray]:
    print("=== Load train X / y ===")
    Xdf = pl.read_parquet(X_TRAIN_PATH)
    ydf = pl.read_parquet(Y_TRAIN_PATH)

    if ID_COL in Xdf.columns:
        print("⚠️ Found 'id' in X -> dropping it for training.")
        Xdf = Xdf.drop(ID_COL)

    Xdf = Xdf.select(_numeric_cols(Xdf))
    X = Xdf.to_numpy().astype(np.float32, copy=False)
    y = ydf.select(TARGET).to_numpy().ravel().astype(np.float32)

    print(f"X_train: {X.shape} | y: {y.shape}")
    return X, y

def load_test() -> tuple[np.ndarray, np.ndarray]:
    print("=== Load test ids + X_test_clean ===")
    raw = pl.read_csv(TEST_RAW_CSV)
    ids = raw.select(ID_COL).to_numpy().ravel()

    Xdf = pl.read_parquet(X_TEST_PATH)
    if ID_COL in Xdf.columns:
        Xdf = Xdf.drop(ID_COL)
    Xdf = Xdf.select(_numeric_cols(Xdf))
    X_test = Xdf.to_numpy().astype(np.float32, copy=False)

    print(f"X_test: {X_test.shape} | ids: {ids.shape}")
    return ids, X_test


# =========================
# LightGBM params
# =========================
def base_params(seed: int) -> dict:
    # tes params “bons”
    return dict(
        objective="regression",
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
        force_col_wise=True,
        num_threads=-1,
    )

def res_params(seed: int) -> dict:
    # plus conservateur : corrige sans overfit
    return dict(
        objective="regression",
        metric="l2",
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=300,
        reg_lambda=10.0,
        reg_alpha=1.0,
        max_depth=-1,
        min_gain_to_split=0.0,
        verbosity=-1,
        seed=seed,
        force_col_wise=True,
        num_threads=-1,
    )


# =========================
# CV runner: OOF + test preds
# =========================
def cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    seeds: list[int],
    params_fn,
    num_boost_round: int,
    early_stop_rounds: int,
    tag: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      - oof_pred (n_train,)
      - test_pred (n_test,)
      - mean_best_iter (int)
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KF_SEED)

    oof_seeds = []
    test_seeds = []
    best_iters_all = []

    for s in seeds:
        print(f"\n[{tag}] === Seed {s} ===")
        oof = np.zeros(len(y), dtype=np.float32)
        test_folds = []

        for fold, (tr, va) in enumerate(kf.split(X), 1):
            dtr = lgb.Dataset(X[tr], label=y[tr], free_raw_data=False)
            dva = lgb.Dataset(X[va], label=y[va], reference=dtr, free_raw_data=False)

            model = lgb.train(
                params_fn(s),
                dtr,
                valid_sets=[dva],
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)],
            )

            best_it = int(model.best_iteration)
            best_iters_all.append(best_it)

            pred_va = model.predict(X[va], num_iteration=best_it).astype(np.float32)
            pred_te = model.predict(X_test, num_iteration=best_it).astype(np.float32)

            oof[va] = pred_va
            test_folds.append(pred_te)

            acc_fold = acc_within_tol(y[va], pred_va, TOL)
            print(f"[{tag} seed {s} fold {fold}] best_iter={best_it} acc@0.05={acc_fold:.4f}")

        oof_seeds.append(oof)
        test_seeds.append(np.mean(np.vstack(test_folds), axis=0).astype(np.float32))

        print(f"[{tag} seed {s}] OOF acc@0.05 = {acc_within_tol(y, oof, TOL):.4f}")

    oof_ens = np.mean(np.vstack(oof_seeds), axis=0).astype(np.float32)
    test_ens = np.mean(np.vstack(test_seeds), axis=0).astype(np.float32)
    mean_best_iter = int(np.round(np.mean(best_iters_all)))

    print(f"\n[{tag}] Ensemble OOF acc@0.05 = {acc_within_tol(y, oof_ens, TOL):.4f}")
    print(f"[{tag}] Mean best_iter = {mean_best_iter}")

    return oof_ens, test_ens, mean_best_iter


# =========================
# Full refit helpers (train on all data)
# =========================
def fit_full_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    seeds: list[int],
    params_fn,
    num_boost_round: int,
    tag: str
) -> np.ndarray:
    dfull = lgb.Dataset(X, label=y, free_raw_data=False)
    preds = []
    for s in seeds:
        model = lgb.train(params_fn(s), dfull, num_boost_round=num_boost_round, valid_sets=None)
        preds.append(model.predict(X_test).astype(np.float32))
    pred = np.mean(np.vstack(preds), axis=0).astype(np.float32)
    print(f"[{tag}] Full-refit test pred done. shape={pred.shape}")
    return pred


# =========================
# Main
# =========================
def main():
    X, y = load_train()
    ids, X_test = load_test()

    # ========= STAGE 1: BASE =========
    print("\n==============================")
    print("=== STAGE 1: Base LGB ensemble ===")
    base_oof, base_test, base_mean_it = cv_predict(
        X=X, y=y, X_test=X_test,
        seeds=BASE_SEEDS,
        params_fn=base_params,
        num_boost_round=BASE_NUM_BOOST,
        early_stop_rounds=BASE_EARLY_STOP,
        tag="BASE"
    )

    base_shift, base_oof_shifted_acc = best_shift_for_metric(y, base_oof, TOL, SHIFT_RANGE, SHIFT_STEPS)
    print(f"\n[BASE] best shift={base_shift:+.5f} -> OOF acc@0.05={base_oof_shifted_acc:.4f}")

    # ========= STAGE 2: RESIDUAL =========
    print("\n==============================")
    print("=== STAGE 2: Residual LGB ensemble ===")

    # résidu à apprendre sur train (cible du 2e étage)
    base_oof_shifted = base_oof + base_shift
    resid_y = (y - base_oof_shifted).astype(np.float32)

    # Features du résiduel : [X, base_pred]
    X_res = np.hstack([X, base_oof_shifted.reshape(-1, 1)]).astype(np.float32)
    X_test_res_input = np.hstack([X_test, (base_test + base_shift).reshape(-1, 1)]).astype(np.float32)

    res_oof, res_test, res_mean_it = cv_predict(
        X=X_res, y=resid_y, X_test=X_test_res_input,
        seeds=RES_SEEDS,
        params_fn=res_params,
        num_boost_round=RES_NUM_BOOST,
        early_stop_rounds=RES_EARLY_STOP,
        tag="RES"
    )

    # ========= COMBINE OOF =========
    oof_final = clip01(base_oof_shifted + res_oof)
    acc_final_raw = acc_within_tol(y, oof_final, TOL)
    final_shift, acc_final_shift = best_shift_for_metric(y, oof_final, TOL, SHIFT_RANGE, SHIFT_STEPS)

    print("\n==============================")
    print("=== OOF SUMMARY ===")
    print(f"Base OOF acc (raw)            : {acc_within_tol(y, base_oof, TOL):.4f}")
    print(f"Base OOF acc (shifted)        : {base_oof_shifted_acc:.4f} (shift={base_shift:+.5f})")
    print(f"Final OOF acc (base+res, raw)  : {acc_final_raw:.4f}")
    print(f"Final OOF acc (after shift)    : {acc_final_shift:.4f} (shift={final_shift:+.5f})")

    # ========= TRAIN FULL FOR KAGGLE =========
    # On refit sur tout le train avec nb itérations = mean_best_iter (même logique que ton script)
    print("\n==============================")
    print("=== FULL REFIT + TEST PRED (for Kaggle) ===")

    base_test_full = fit_full_and_predict(X, y, X_test, BASE_SEEDS, base_params, base_mean_it, tag="BASE-FULL")
    base_test_full = base_test_full + base_shift  # même shift que CV

    # Refit residual on full: target = y - base_pred_full_train (mais on ne peut pas avoir base_pred_full_train sans fuite)
    # => méthode standard stacking: on entraine residual sur OOF (déjà fait) et on refit residual sur full en utilisant
    # la base prédite sur full pour le train (c'est acceptable pratique Kaggle, mais attention risque léger de surfit).
    print("\n[RES-FULL] Build residual target with base full prediction on train...")
    base_train_full_pred = fit_full_and_predict(X, y, X, BASE_SEEDS, base_params, base_mean_it, tag="BASE-FULL-ON-TRAIN")
    base_train_full_pred = base_train_full_pred + base_shift
    resid_full_y = (y - clip01(base_train_full_pred)).astype(np.float32)

    X_res_full = np.hstack([X, clip01(base_train_full_pred).reshape(-1, 1)]).astype(np.float32)
    X_test_res_full = np.hstack([X_test, clip01(base_test_full).reshape(-1, 1)]).astype(np.float32)

    res_test_full = fit_full_and_predict(X_res_full, resid_full_y, X_test_res_full, RES_SEEDS, res_params, res_mean_it, tag="RES-FULL")

    test_pred = clip01(base_test_full + res_test_full)
    test_pred = clip01(test_pred + final_shift)

    # ========= SUBMISSION =========
    sub = pl.DataFrame({
        "id": ids,
        "wip": np.zeros(len(ids), dtype=np.float32),
        "investissement": np.zeros(len(ids), dtype=np.float32),
        "satisfaction": test_pred.astype(np.float32),
    })

    out_csv = ART / "submission_test_ensemble+LGBM.csv"
    sub.write_csv(out_csv)
    print(f"\n✅ Saved submission: {out_csv}")

    # ========= REPORT =========
    report = "\n".join([
        "=== test_ensemble+LGBM REPORT ===",
        f"BASE_SEEDS={BASE_SEEDS}",
        f"RES_SEEDS={RES_SEEDS}",
        f"base_mean_best_iter={base_mean_it}",
        f"res_mean_best_iter={res_mean_it}",
        f"base_shift={base_shift:.8f}",
        f"final_shift={final_shift:.8f}",
        f"base_oof_acc_raw={acc_within_tol(y, base_oof, TOL):.6f}",
        f"base_oof_acc_shifted={base_oof_shifted_acc:.6f}",
        f"final_oof_acc_raw={acc_final_raw:.6f}",
        f"final_oof_acc_shifted={acc_final_shift:.6f}",
    ])
    (ART / "test_ensemble+LGBM_report.txt").write_text(report, encoding="utf-8")
    print(f"✅ Saved report: {ART/'test_ensemble+LGBM_report.txt'}")


if __name__ == "__main__":
    main()
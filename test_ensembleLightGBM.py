#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

# =========================
# Paths / config
# =========================
DATA = Path("data")
ART  = Path("artifacts")

TEST_RAW_PATH = DATA / "test.csv"              # contient id
X_TEST_PATH   = DATA / "X_test_clean.parquet"  # features cleaned (sans id)

ID_COL = "id"
SEEDS = [1, 2, 3, 4, 5]

OUT_PATH = ART / "submission.csv"

# =========================
# Utils
# =========================
def is_numeric_polars(dt: pl.DataType) -> bool:
    return dt in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )

# =========================
# Main
# =========================
def main():
    ART.mkdir(exist_ok=True)

    print("=== Load test ids ===")
    test_raw = pl.read_csv(TEST_RAW_PATH)
    ids = test_raw.select(ID_COL).to_numpy().ravel()

    print("=== Load X_test_clean.parquet ===")
    Xdf = pl.read_parquet(X_TEST_PATH)

    if ID_COL in Xdf.columns:
        print("⚠️ Found 'id' in X_test_clean -> dropping it.")
        Xdf = Xdf.drop(ID_COL)

    numeric_cols = [c for c, dt in zip(Xdf.columns, Xdf.dtypes) if is_numeric_polars(dt)]
    Xdf = Xdf.select(numeric_cols)
    X_test = Xdf.to_numpy().astype(np.float32, copy=False)

    print(f"X_test shape: {X_test.shape}")

    # === Load shift ===
    shift_path = ART / "calibration_shift_satisfaction.txt"
    shift = float(shift_path.read_text().strip())
    print(f"Using shift: {shift:+.8f}")

    # === Load models & predict ===
    preds = []
    for s in SEEDS:
        model_path = ART / f"lgbm_satisfaction_full_seed{s}.txt"
        model = lgb.Booster(model_file=str(model_path))
        preds.append(model.predict(X_test).astype(np.float32))

    pred_satisfaction = np.mean(np.vstack(preds), axis=0) + shift

    # === Build submission ===
    submission = pl.DataFrame({
        "id": ids,
        "wip": np.zeros_like(pred_satisfaction),
        "investissement": np.zeros_like(pred_satisfaction),
        "satisfaction": pred_satisfaction,
    })

    submission.write_csv(OUT_PATH)

    print(f"\n✅ Submission saved: {OUT_PATH}")
    print("Format: id,wip,investissement,satisfaction")
    print("wip & investissement = 0 everywhere")

if __name__ == "__main__":
    main()
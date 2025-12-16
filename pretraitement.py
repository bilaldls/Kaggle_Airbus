from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Paths / config
# =========================
DATA = Path("data")
TRAIN_PATH = DATA / "train.csv"
TEST_PATH  = DATA / "test.csv"

OUT_DIR = DATA
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["wip", "investissement", "satisfaction"]
ID_COL = "id"

CORR_THRESHOLD = 0.95
BLOCK_SIZE = 500
NEAR_CONSTANT_THRESHOLD = 1

COL_TXT = DATA / "columns_kept.txt"
COL_CSV = DATA / "columns_kept.csv"

XTRAIN_OUT = DATA / "X_train_clean.parquet"
XTEST_OUT  = DATA / "X_test_clean.parquet"
YTRAIN_OUT = DATA / "y_train.parquet"


# =========================
# Utils
# =========================
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
    return df


def drop_useless_columns(
    df: pd.DataFrame,
    *,
    exclude: set[str],
    near_constant_threshold: float
) -> tuple[pd.DataFrame, dict]:

    info = {"all_missing": [], "constant": [], "near_constant": []}
    cols = [c for c in df.columns if c not in exclude]

    # all missing
    all_missing = [c for c in cols if df[c].isna().all()]
    df = df.drop(columns=all_missing)
    info["all_missing"] = all_missing

    cols = [c for c in df.columns if c not in exclude]

    # constant
    nunique = df[cols].nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant)
    info["constant"] = constant

    cols = [c for c in df.columns if c not in exclude]

    # near constant
    near_constant = []
    n = len(df)
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        if len(vc) > 0 and (vc.iloc[0] / n) >= near_constant_threshold:
            near_constant.append(c)

    df = df.drop(columns=near_constant)
    info["near_constant"] = near_constant

    return df, info


def drop_collinear_features(
    df: pd.DataFrame,
    *,
    exclude: set[str],
    threshold: float,
    block_size: int
) -> tuple[pd.DataFrame, list[str]]:

    numeric_cols = [
        c for c in df.columns
        if (
            c not in exclude
            and c != ID_COL
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    print(f"\n[Collinearity] Analysing {len(numeric_cols)} numeric columns...")
    to_drop = set()

    work_df = df[numeric_cols].copy()
    work_df = work_df.fillna(work_df.median(numeric_only=True))

    for i in range(0, len(numeric_cols), block_size):
        block_i = numeric_cols[i:i + block_size]
        data_i = work_df[block_i].values.T

        for j in range(i + block_size, len(numeric_cols), block_size):
            block_j = numeric_cols[j:j + block_size]
            data_j = work_df[block_j].values.T

            corr = np.abs(np.corrcoef(data_i, data_j)[:len(block_i), len(block_i):])

            for idx_i, col_i in enumerate(block_i):
                if col_i in to_drop:
                    continue
                for idx_j, col_j in enumerate(block_j):
                    if col_j in to_drop:
                        continue
                    if corr[idx_i, idx_j] >= threshold:
                        to_drop.add(col_j)

    df = df.drop(columns=list(to_drop))
    return df, sorted(to_drop)


# =========================
# Main
# =========================
def main():
    print("=== Load train.csv ===")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = optimize_dtypes(train)
    test  = optimize_dtypes(test)

    exclude = set(TARGETS + [ID_COL])

    # ---------- Train-only preprocessing ----------
    train_clean, drop_info = drop_useless_columns(
        train, exclude=exclude, near_constant_threshold=NEAR_CONSTANT_THRESHOLD
    )

    print("\n=== Cheap cleaning summary ===")
    for k, v in drop_info.items():
        print(f"{k}: {len(v)}")

    train_clean, dropped_collinear = drop_collinear_features(
        train_clean,
        exclude=exclude,
        threshold=CORR_THRESHOLD,
        block_size=BLOCK_SIZE
    )

    print(f"[Collinearity] Dropped: {len(dropped_collinear)}")

    # ---------- Build feature list ----------
    feature_cols = [c for c in train_clean.columns if c not in exclude and c != ID_COL]
    print(f"\nKept features: {len(feature_cols)}")

    # ---------- Save columns ----------
    COL_TXT.write_text("\n".join(feature_cols), encoding="utf-8")
    pd.Series(feature_cols, name="feature").to_csv(COL_CSV, index=False)

    print(f"Saved column list:")
    print(f" - {COL_TXT}")
    print(f" - {COL_CSV}")

    # ---------- Build final datasets ----------
    X_train = train_clean[feature_cols]
    y_train = train_clean[TARGETS]

    X_test = test.copy()
    X_test = X_test[feature_cols]  # same columns, same order

    # ---------- Save datasets ----------
    X_train.to_parquet(XTRAIN_OUT, index=False)
    X_test.to_parquet(XTEST_OUT, index=False)
    y_train.to_parquet(YTRAIN_OUT, index=False)

    print("\nSaved datasets:")
    print(f" - {XTRAIN_OUT}")
    print(f" - {XTEST_OUT}")
    print(f" - {YTRAIN_OUT}")

    print("\n✅ Preprocessing finished successfully.")


if __name__ == "__main__":
    main()
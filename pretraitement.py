#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/train.csv")
TEST_PATH  = Path("data/test.csv")

TARGETS = ["wip", "investissement", "satisfaction"]
ID_COL = "id"

CORR_THRESHOLD = 0.95   # seuil de colinéarité
BLOCK_SIZE = 500        # taille de bloc pour corrélation

# ---------------------------------------------------
# Utils
# ---------------------------------------------------

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
    near_constant_threshold: float = 0.999
) -> tuple[pd.DataFrame, dict]:

    info = {
        "all_missing": [],
        "constant": [],
        "near_constant": [],
    }

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
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        if len(vc) > 0 and vc.iloc[0] / len(df) >= near_constant_threshold:
            near_constant.append(c)

    df = df.drop(columns=near_constant)
    info["near_constant"] = near_constant

    return df, info


def drop_collinear_features(
    df: pd.DataFrame,
    *,
    exclude: set[str],
    threshold: float = 0.95,
    block_size: int = 500
) -> tuple[pd.DataFrame, list[str]]:
    """
    Supprime les colonnes numériques fortement colinéaires.
    - id est EXCLU explicitement
    - garde une seule colonne par groupe colinéaire
    """

    numeric_cols = [
        c for c in df.columns
        if (
            c not in exclude
            and c != ID_COL                # sécurité explicite
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    print(f"\n[Collinearity] Analysing {len(numeric_cols)} numeric columns...")

    to_drop = set()

    # Remplacer NaN → médiane (nécessaire pour corrcoef)
    work_df = df[numeric_cols].copy()
    work_df = work_df.fillna(work_df.median(numeric_only=True))

    for i in range(0, len(numeric_cols), block_size):
        block_i = numeric_cols[i:i + block_size]
        data_i = work_df[block_i].values.T

        for j in range(i + block_size, len(numeric_cols), block_size):
            block_j = numeric_cols[j:j + block_size]
            data_j = work_df[block_j].values.T

            corr = np.abs(
                np.corrcoef(data_i, data_j)[:len(block_i), len(block_i):]
            )

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
# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("=== Loading train.csv ===")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Raw train shape: {train.shape}")

    train = optimize_dtypes(train)

    exclude = set(TARGETS + [ID_COL])

    # Step 1a — cheap cleaning
    train, drop_info = drop_useless_columns(train, exclude=exclude)

    print("\n=== Cheap cleaning summary ===")
    for k, v in drop_info.items():
        print(f"{k}: {len(v)}")

    print(f"After cheap cleaning: {train.shape}")

    # Step 1b — collinearity
    train, dropped_collinear = drop_collinear_features(
        train,
        exclude=exclude,
        threshold=CORR_THRESHOLD,
        block_size=BLOCK_SIZE
    )

    print(f"\n[Collinearity] Dropped columns: {len(dropped_collinear)}")
    print(f"After collinearity: {train.shape}")

    # Split
    y = train[TARGETS].copy()
    X = train.drop(columns=TARGETS)

    print("\n=== Memory report ===")
    mem_mb = train.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"train memory: {mem_mb:.1f} MB")

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    X.to_parquet(out_dir / "X_train_clean.parquet", index=False)
    y.to_parquet(out_dir / "y_train.parquet", index=False)

    print("\nSaved cleaned train datasets")

if __name__ == "__main__":
    main()
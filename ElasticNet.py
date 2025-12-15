#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

ART = Path("data")
X_PATH = ART / "X_train_clean.parquet"
Y_PATH = ART / "y_train.parquet"

ID_COL = "id"
TARGET = "satisfaction"

# ---- selection policy (not aggressive)
TOP_CORR = 4500          # large prefilter (your current X has 5532 cols)
SAMPLE_ROWS = 40000      # fit faster; increase to 60000 if ok
RANDOM_STATE = 42

# ---- elasticnet "soft" (close to Ridge)
ALPHA = 3e-5             # weak regularization
L1_RATIO = 0.2           # low sparsity pressure
EPOCHS = 50

# ---- final keep rule (quantile on |coef|)
KEEP_QUANTILE = 0.70     # keep top 30% by |coef| among prefiltered

def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )

def main():
    print("=== Load parquet (polars) ===")
    X = pl.read_parquet(X_PATH)
    ydf = pl.read_parquet(Y_PATH)

    if TARGET not in ydf.columns:
        raise ValueError(f"Target '{TARGET}' not in {Y_PATH}")

    y = ydf.select(TARGET).to_numpy().ravel().astype(np.float32)

    numeric_features = [
        name for name, dtype in zip(X.columns, X.dtypes)
        if name != ID_COL and is_numeric_dtype(dtype)
    ]
    print(f"Numeric features: {len(numeric_features)}")

    Xn = X.select(numeric_features)

    # median impute
    meds = Xn.select([pl.median(c).alias(c) for c in numeric_features]).row(0)
    med_map = dict(zip(numeric_features, meds))
    Xn = Xn.with_columns([pl.col(c).fill_null(med_map[c]) for c in numeric_features])

    # ---- broad corr prefilter (fast enough with 5.5k cols)
    print(f"=== Pre-filter corr: keep top {TOP_CORR} ===")
    y_pl = pl.Series(TARGET, y)

    corrs = []
    for c in numeric_features:
        v = Xn.select(pl.corr(pl.col(c), y_pl).abs()).item()
        if v is None or np.isnan(v):
            v = 0.0
        corrs.append((c, float(v)))

    corrs.sort(key=lambda t: t[1], reverse=True)
    kept = [c for c, _ in corrs[:min(TOP_CORR, len(corrs))]]
    print(f"Kept after corr filter: {len(kept)}")

    # ---- sample rows for speed
    n = Xn.height
    if SAMPLE_ROWS is not None and SAMPLE_ROWS < n:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=SAMPLE_ROWS, replace=False)
        idx.sort()
        X_fit = Xn.select(kept).to_numpy()[idx]
        y_fit = y[idx]
        print(f"Fitting on sample rows: {SAMPLE_ROWS}/{n}")
    else:
        X_fit = Xn.select(kept).to_numpy()
        y_fit = y
        print(f"Fitting on full rows: {n}")

    # ---- standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)

    # ---- soft elasticnet via SGD
    print("=== Fit SGDRegressor (soft elasticnet) ===")
    model = SGDRegressor(
        loss="squared_error",
        penalty="elasticnet",
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        max_iter=EPOCHS,
        tol=1e-3,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(Xs, y_fit)

    abs_coef = np.abs(model.coef_)
    # Keep features above a quantile of |coef| (not too aggressive)
    thr = float(np.quantile(abs_coef, KEEP_QUANTILE))
    selected = [f for f, c in zip(kept, abs_coef) if c >= thr]

    print(f"Quantile threshold (|coef|) @ {KEEP_QUANTILE:.2f}: {thr:.6g}")
    print(f"Selected features: {len(selected)} (from {len(kept)})")

    # save list
    out_txt = ART / f"selected_features_soft_{len(selected)}.txt"
    out_txt.write_text("\n".join(selected), encoding="utf-8")
    print(f"Saved: {out_txt}")

    # save reduced parquet (keep id + selected)
    if ID_COL in X.columns:
        X_sel = X.select([ID_COL] + selected)
    else:
        X_sel = X.select(selected)

    out_parquet = ART / "X_train_selected_soft.parquet"
    X_sel.write_parquet(out_parquet)
    print(f"Saved: {out_parquet}")

if __name__ == "__main__":
    main()
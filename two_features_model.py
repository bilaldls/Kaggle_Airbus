import gc
import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool

N_FOLDS = 5
RANDOM_STATE = 42
ID_COL = "id"
PREVIOUS_SUBMISSION = "submission_final_stacking.csv"  # wip + investissement

#load
X_train = pl.read_parquet("data/X_train_clean.parquet").to_pandas()
y_train = pl.read_parquet("data/y_train.parquet").to_pandas()
X_test  = pl.read_parquet("data/X_test_clean.parquet").to_pandas()

prev_sub = pd.read_csv(PREVIOUS_SUBMISSION)

id_test = prev_sub[ID_COL]
wip_pred = prev_sub["wip"].values
invest_pred = prev_sub["investissement"].values


#features separation
demand_cols = [c for c in X_train.columns if c.startswith("demand_")]
param_cols  = [c for c in X_train.columns if c.startswith("param_")]

X_demand_tr = X_train[demand_cols].to_numpy(dtype=np.float32)
X_demand_te = X_test[demand_cols].to_numpy(dtype=np.float32)

X_param_tr = X_train[param_cols].to_numpy(dtype=np.float32)
X_param_te = X_test[param_cols].to_numpy(dtype=np.float32)


#scaling target
scaler = MinMaxScaler()
y_sat = scaler.fit_transform(y_train[["satisfaction"]]).ravel()

#KFold
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_demand = np.zeros(len(X_train), dtype=np.float32)
oof_param  = np.zeros(len(X_train), dtype=np.float32)

test_demand = np.zeros(len(X_test), dtype=np.float32)
test_param  = np.zeros(len(X_test), dtype=np.float32)


#CV
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_demand_tr)):
    print(f"\nFold {fold + 1}")

    #demand model
    cat = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.1,
        loss_function="RMSE",
        od_type="Iter",
        od_wait=50,
        random_seed=RANDOM_STATE,
        task_type="CPU",
        thread_count=4,
        verbose=False
    )

    cat.fit(
        X_demand_tr[tr_idx],
        y_sat[tr_idx],
        eval_set=Pool(X_demand_tr[val_idx], y_sat[val_idx])
    )

    oof_demand[val_idx] = cat.predict(X_demand_tr[val_idx])
    test_demand += cat.predict(X_demand_te)

    #parameter model
    ridge_param = Ridge(alpha=1.0)

    ridge_param.fit(X_param_tr[tr_idx], y_sat[tr_idx])

    oof_param[val_idx] = ridge_param.predict(X_param_tr[val_idx])
    test_param += ridge_param.predict(X_param_te)

    #for RAM memory management
    del cat, ridge_param
    gc.collect()


#average test predictions
test_demand /= N_FOLDS
test_param  /= N_FOLDS


#stacking
oof_stack  = np.column_stack([oof_demand, oof_param])
test_stack = np.column_stack([test_demand, test_param])

final_ridge = Ridge(alpha=1.0)
final_ridge.fit(oof_stack, y_sat)

final_sat_scaled = final_ridge.predict(test_stack)
final_satisfaction = scaler.inverse_transform(
    final_sat_scaled.reshape(-1, 1)
).ravel()


#submission
submission = pd.DataFrame({
    "id": id_test,
    "wip": wip_pred,
    "investissement": invest_pred,
    "satisfaction": final_satisfaction
})

submission.to_csv("submission_satisfaction_demand_param.csv", index=False)
print("submission_satisfaction_demand_param.csv generated")

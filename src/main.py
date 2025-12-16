from data.preprocess import PreprocessConfig, load_raw_data, preprocess_fit_transform
from models.multi_output_MLP import ModelConfig, AdvancedMultiOutputRegressor

# 1) Preprocessing
pcfg = PreprocessConfig(train_path="data/raw/train.csv", test_path="data/raw/test.csv")
train_df, test_df = load_raw_data(pcfg)
X_train, y_train, X_val, y_val, X_test, preproc, train_ids, val_ids, test_ids = preprocess_fit_transform(train_df, test_df, pcfg)

# 2) Model
mcfg = ModelConfig(input_dim=X_train.shape[1], device="cpu")
model = AdvancedMultiOutputRegressor(mcfg)
model.fit(X_train, y_train, X_val, y_val)

# 3) Predict (calibrated)
y_test_pred = model.predict(X_test)  # shape (N,3)

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load Data to Score (e.g., test.csv or any new production batch)
data_path = PROJECT_ROOT / "data" / "test.csv"
df_new = pd.read_csv(data_path)
ids = df_new["id"]
X_raw = df_new.drop(columns=["id"], errors="ignore")

# 2. LightGBM Inference (Native Categorical Dtypes)
X_lgb = X_raw.copy()
cat_cols = X_lgb.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
  X_lgb[col] = X_lgb[col].astype("category")

lgb_preds = np.zeros(len(df_new))
for fold in range(1, 6):
  model = joblib.load(PROJECT_ROOT / f"models/lgbm_fold_{fold}.joblib")
  lgb_preds += model.predict_proba(X_lgb)[:, 1] / 5.0

# 3. XGBoost Inference (ColumnTransformer + Scaler)
preprocessor = joblib.load(PROJECT_ROOT / "models/xgb_preprocessor.joblib")
X_xgb_proc = preprocessor.transform(X_raw)

xgb_preds = np.zeros(len(df_new))
for fold in range(1, 6):
  model = joblib.load(PROJECT_ROOT / f"models/xgb_fold_{fold}.joblib")
  xgb_preds += model.predict_proba(X_xgb_proc)[:, 1] / 5.0

# 4. Apply Winning Blend Weights (0.68 LGBM + 0.32 XGBoost)
final_probs = (0.68 * lgb_preds) + (0.32 * xgb_preds)

# 5. Output Scored Results
output_df = pd.DataFrame({"id": ids, "loan_default_probability": final_probs})
output_path = PROJECT_ROOT / "data" / "scored_predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"Scored {len(df_new):,} records successfully.")
print(f"Output saved to: {output_path}")
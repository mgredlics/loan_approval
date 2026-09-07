import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Parse Command-Line Arguments
parser = argparse.ArgumentParser(
    description="Score applicant batches using champion ensemble."
)
parser.add_argument(
    "--input",
    type=str,
    default="data/test.csv",
    help="Relative path to raw CSV file to score",
)
parser.add_argument(
    "--output",
    type=str,
    default="data/scored_predictions.csv",
    help="Relative path for scored output CSV",
)
args = parser.parse_args()

# 2. Load Target Dataset
data_path = PROJECT_ROOT / args.input
print(f"Loading data from: {data_path}")

df_new = pd.read_csv(data_path)
ids = (
    df_new["id"] if "id" in df_new.columns else pd.Series(range(len(df_new)))
)
X_raw = df_new.drop(columns=["id", "loan_status"], errors="ignore")

# 3. LightGBM Fold Inference
X_lgb = X_raw.copy()
cat_cols = X_lgb.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
  X_lgb[col] = X_lgb[col].astype("category")

lgb_preds = np.zeros(len(df_new))
for fold in range(1, 6):
  model = joblib.load(PROJECT_ROOT / f"models/lgbm_fold_{fold}.joblib")
  lgb_preds += model.predict_proba(X_lgb)[:, 1] / 5.0

# 4. XGBoost Fold Inference
preprocessor = joblib.load(PROJECT_ROOT / "models/xgb_preprocessor.joblib")
X_xgb_proc = preprocessor.transform(X_raw)

xgb_preds = np.zeros(len(df_new))
for fold in range(1, 6):
  model = joblib.load(PROJECT_ROOT / f"models/xgb_fold_{fold}.joblib")
  xgb_preds += model.predict_proba(X_xgb_proc)[:, 1] / 5.0

# 5. Champion Blend (0.68 LGBM + 0.32 XGB)
final_probs = (0.68 * lgb_preds) + (0.32 * xgb_preds)

output_df = pd.DataFrame({"id": ids, "loan_default_probability": final_probs})
output_path = PROJECT_ROOT / args.output
output_df.to_csv(output_path, index=False)

print(f"Scored {len(df_new):,} records successfully -> {output_path}")
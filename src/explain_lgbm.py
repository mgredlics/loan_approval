import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 1. Load Data & Prepare Types
train_df = pd.read_csv("data/train.csv")
TARGET = "loan_status"

X = train_df.drop(columns=[TARGET, "id"])
y = train_df[TARGET]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype("category")

# 2. Fit a Model on Representative Data
print("Training LightGBM model for SHAP explanations...")
model = LGBMClassifier(
    n_estimators=450,
    learning_rate=0.04,
    num_leaves=45,
    min_child_samples=25,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.80,
    reg_alpha=0.5,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
model.fit(X, y)

# 3. Sample for Rapid Computation (2,500 rows)
X_sample = X.sample(n=2500, random_state=42)

# 4. TreeExplainer on LightGBM
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# LightGBM binary classification can return values for [Class 0, Class 1]
# Slice to Class 1 (Default probability) if returned as 3D
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# 5. Generate and Save the Beeswarm Plot
Path("reports").mkdir(exist_ok=True)
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("LightGBM SHAP Beeswarm: Unified Feature Impact", fontsize=13, weight="bold", pad=15)
plt.tight_layout()

output_path = "reports/lgbm_shap_beeswarm.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved SHAP beeswarm plot to {output_path}")
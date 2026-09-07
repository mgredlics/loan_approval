import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import mlflow
import os

# 1. Load Ground Truth and Saved Out-Of-Fold Probabilities
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
y_true = train_df["loan_status"]

oof_xgb = np.load("data/oof_xgb.npy")
oof_lgb = np.load("data/oof_lgbm.npy")
test_xgb = np.load("data/test_preds_xgb.npy")
test_lgb = np.load("data/test_preds_lgbm.npy")

print(f"XGBoost Standalone OOF AUC:  {roc_auc_score(y_true, oof_xgb):.5f}")
print(f"LightGBM Standalone OOF AUC: {roc_auc_score(y_true, oof_lgb):.5f}")
print("-" * 50)

# 2. Grid Search the Optimal Blend Weight
best_weight = 0.5
best_auc = 0.0

for w in np.linspace(0.0, 1.0, 101):
    blended_oof = (w * oof_lgb) + ((1.0 - w) * oof_xgb)
    score = roc_auc_score(y_true, blended_oof)
    if score > best_auc:
        best_auc = score
        best_weight = w

print(f"Optimal Weight: {best_weight:.2f} * LightGBM + {1 - best_weight:.2f} * XGBoost")
print(f"Blended Ensemble OOF AUC:   {best_auc:.5f}")
print(f"Net Lift Over Single Best:  {best_auc - max(roc_auc_score(y_true, oof_xgb), roc_auc_score(y_true, oof_lgb)):+.5f}")

# 3. Create Ensemble Submission
blended_test = (best_weight * test_lgb) + ((1.0 - best_weight) * test_xgb)
pd.DataFrame({
    "id": test_df["id"],
    "loan_status": blended_test
}).to_csv("data/submission_ensemble_blend.csv", index=False)

print("\nSaved blended submission to data/submission_ensemble_blend.csv")

# Log Champion Ensemble to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Competition")

with mlflow.start_run(run_name="champion_ensemble_blend"):
    # 1. Log Blend Weights and Inputs
    mlflow.log_params({
        "model_type": "WeightedEnsemble",
        "weight_lgbm": round(best_weight, 2),
        "weight_xgboost": round(1.0 - best_weight, 2),
        "base_models": "Tuned_XGBoost + LightGBM_NativeCat"
    })
    
    # 2. Log Performance Metrics
    mlflow.log_metrics({
        "oof_roc_auc": best_auc,
        "lgbm_standalone_auc": roc_auc_score(y_true, oof_lgb),
        "xgb_standalone_auc": roc_auc_score(y_true, oof_xgb),
        "net_lift_over_best": best_auc - max(roc_auc_score(y_true, oof_xgb), roc_auc_score(y_true, oof_lgb))
    })
    
    # 3. Log Artifacts (Submission CSV and SHAP Diagnostic)
    mlflow.log_artifact("data/submission_ensemble_blend.csv", artifact_path="submissions")
    
    shap_img = "reports/lgbm_shap_beeswarm.png"
    if os.path.exists(shap_img):
        mlflow.log_artifact(shap_img, artifact_path="diagnostics")

print("Logged champion ensemble and artifacts to MLflow.")
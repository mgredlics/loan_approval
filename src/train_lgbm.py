import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 1. Load Data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

TARGET = "loan_status"
X = train_df.drop(columns=[TARGET, "id"])
y = train_df[TARGET]
X_test = test_df.drop(columns=["id"])

# 2. Native Categorical Typing
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 3. Model Configuration
# LightGBM parameterizes tree capacity via 'num_leaves' rather than 'max_depth'
lgbm_params = {
    "n_estimators": 450,
    "learning_rate": 0.04,
    "num_leaves": 45,            # Controls leaf capacity (roughly equivalent to depth 6-7)
    "min_child_samples": 25,     # LightGBM's equivalent to min_child_weight
    "subsample": 0.85,
    "subsample_freq": 1,         # Required in LGBM to activate subsampling
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_jobs": 4,
    "verbose": -1
}

# 4. Stratified 5-Fold Cross-Validation
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Competition")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
fold_aucs = []

with mlflow.start_run(run_name="challenger_lightgbm_5fold"):
    mlflow.log_params(lgbm_params)
    print("Beginning 5-Fold Stratified CV with LightGBM (Native Categoricals)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[]
        )

        # Ensure local models directory exists
        Path("models").mkdir(exist_ok=True)
        
        # Save model to disk and log as artifact
        model_path = f"models/lgbm_fold_{fold}.joblib"
        import joblib
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc)
        print(f"  Fold {fold} ROC-AUC: {fold_auc:.5f}")

    final_lgb_auc = roc_auc_score(y, oof_preds)
    mlflow.log_metric("oof_roc_auc", final_lgb_auc)
    mlflow.log_metric("mean_cv_auc", np.mean(fold_aucs))

    print("\n" + "="*50)
    print(f"Tuned XGBoost OOF ROC-AUC:  0.95714")
    print(f"LightGBM OOF ROC-AUC:       {final_lgb_auc:.5f}")
    print(f"Delta (LGBM vs XGBoost):    {(final_lgb_auc - 0.95714):+.5f}")
    print("="*50)

# 5. Save Artifacts for Ensembling
np.save("data/oof_lgbm.npy", oof_preds)
np.save("data/test_preds_lgbm.npy", test_preds)

submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_status": test_preds
})
submission.to_csv("data/submission_lgbm.csv", index=False)
print("Saved predictions to data/submission_lgbm.csv")
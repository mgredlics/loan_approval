import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

TARGET = "loan_status"
X = train_df.drop(columns=[TARGET, "id"])
y = train_df[TARGET]
X_test = test_df.drop(columns=["id"])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_tr_proc = preprocessor.fit_transform(X)
X_te_proc = preprocessor.transform(X_test)

best_params = {
    "n_estimators": 450,
    "learning_rate": 0.0541,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8588,
    "colsample_bytree": 0.8309,
    "reg_alpha": 0.2481,
    "reg_lambda": 3.8343,
    "random_state": 42,
    "eval_metric": "auc",
    "n_jobs": 4
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Competition")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

with mlflow.start_run(run_name="final_tuned_xgboost_5fold"):
    mlflow.log_params(best_params)

    joblib.dump(preprocessor, "models/xgb_preprocessor.joblib")
    mlflow.log_artifact("models/xgb_preprocessor.joblib", artifact_path="models")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr_proc, y), start=1):
        X_train_fold, y_train_fold = X_tr_proc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X_tr_proc[val_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**best_params)
        model.fit(X_train_fold, y_train_fold)

        
        # Ensure local models directory exists
        Path("models").mkdir(exist_ok=True)
        
        # Save model to disk and log as artifact
        model_path = f"models/xgb_fold_{fold}.joblib"
        import joblib
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")
        
        oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        test_preds += model.predict_proba(X_te_proc)[:, 1] / skf.n_splits
        
    final_auc = roc_auc_score(y, oof_preds)
    mlflow.log_metric("oof_roc_auc", final_auc)
    print(f"\nFinal Verified Tuned OOF ROC-AUC: {final_auc:.5f}")

np.save("data/oof_xgb.npy", oof_preds)
np.save("data/test_preds_xgb.npy", test_preds)

submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_status": test_preds
})
submission.to_csv("data/submission_xgb_tuned.csv", index=False)
print("Saved submission to data/submission_xgb_tuned.csv")
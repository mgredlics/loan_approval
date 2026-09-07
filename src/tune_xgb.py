import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# 1. Paths and Data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

TARGET = "loan_status"
X = train_df.drop(columns=[TARGET, "id"])
y = train_df[TARGET]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Pre-transform matrices to avoid re-running ColumnTransformer inside every trial
X_transformed = preprocessor.fit_transform(X)

# 2. MLflow Experiment Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Competition")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Optuna Objective Function
def objective(trial):
    params = {
        "n_estimators": 400,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.65, 0.90),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.85),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "random_state": 42,
        "eval_metric": "auc",
        "n_jobs": 4
    }

    oof_preds = np.zeros(len(train_df))

    for train_idx, val_idx in skf.split(X_transformed, y):
        X_tr, y_tr = X_transformed[train_idx], y.iloc[train_idx]
        X_va, y_va = X_transformed[val_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]

    trial_auc = roc_auc_score(y, oof_preds)

    # Log each trial as a distinct run
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("oof_roc_auc", trial_auc)

    return trial_auc

# 4. Run Study (15 Trials)
print("Starting Bayesian Hyperparameter Search...")
study = optuna.create_study(direction="maximize")

with mlflow.start_run(run_name="xgboost_optuna_tuning_study"):
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    best_auc = study.best_value
    
    mlflow.log_params(best_params)
    mlflow.log_metric("best_oof_roc_auc", best_auc)

    print("\n" + "="*50)
    print(f"Optimization Finished. Best OOF ROC-AUC: {best_auc:.5f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*50)
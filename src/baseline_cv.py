# ==============================================================================
# TABULAR ML PROJECT BOILERPLATE: INGESTION & DATA TRIAGE
# ==============================================================================

# %% [0] Path and Environment Configuration
import os
os.environ["OMP_NUM_THREADS"] = "4"
import warnings
from pathlib import Path

# Equivalent to here::here() in R:
# Resolves the project root regardless of where the kernel was launched
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)

# Suppress standard downstream deprecation noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Working Directory: {PROJECT_ROOT}")

# %% [1] Core Libraries
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

# Scikit-learn essentials
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, classification_report

# Set pandas display options (avoids truncated output in the Interactive Window)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)


# %% [2] Ingestion & Fast Triage Helper Functions
def audit_dataset(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Equivalent to a combined dim() + glimpse() + missingness scan in R."""
    print(f"\n{'='*25} {name.upper()} AUDIT {'='*25}")
    print(f"Dimensions: {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    # Missing value summary
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_table = pd.concat([missing, missing_pct], axis=1, keys=["count", "pct"])
    missing_cols = missing_table[missing_table["count"] > 0]

    if not missing_cols.empty:
        print("\nMissing Features:")
        print(missing_cols.sort_values(by="count", ascending=False))
    else:
        print("\nMissing Features: None (Clean)")

    # Data type summary
    types_count = df.dtypes.value_axis() if hasattr(df.dtypes, "value_axis") else df.dtypes.value_counts()
    print(f"\nColumn Data Types:\n{types_count}")


# Load raw files
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

audit_dataset(train_df, "Train")
audit_dataset(test_df, "Test")


# %% [3] Target & Predictor Dissection
TARGET = "loan_status"
ID_COL = "id" if "id" in train_df.columns else None

# Check class balance (Equivalent to prop.table(table(train_df$target)))
print(f"\n{'='*25} TARGET DISTRIBUTION: '{TARGET}' {'='*25}")
print(train_df[TARGET].value_counts(normalize=True).round(4).to_string())

# Drop ID columns and Target from feature list
exclude_cols = [TARGET] + ([ID_COL] if ID_COL else [])
candidate_features = [col for col in train_df.columns if col not in exclude_cols]

# Programmatic feature typing (Equivalent to recipes selectors)
# numeric_features = all_numeric_predictors()
# categorical_features = all_nominal_predictors()
numeric_features = train_df[candidate_features].select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = train_df[candidate_features].select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nDetected {len(numeric_features)} Numeric Features:\n  {numeric_features}")
print(f"\nDetected {len(categorical_features)} Categorical Features:\n  {categorical_features}")

# Matrices ready for modeling
X = train_df[candidate_features]
y = train_df[TARGET]
X_test = test_df[candidate_features]


# %% [4] Preprocessing Specification
# In R: recipe(loan_status ~ ., data = train) %>% step_normalize() %>% step_dummy()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %% [5] 5-Fold Stratified CV with Logistic Regression Baseline
from sklearn.linear_model import LogisticRegression

# 1. Setup MLflow Experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Competition")

# 2. Setup 5-Fold Stratified Split (Equivalent to rsample::vfold_cv(v=5, strata=loan_status))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store Out-Of-Fold (OOF) and Test Predictions
oof_preds = np.zeros(len(train_df))
test_preds_folds = np.zeros(len(test_df))
fold_aucs = []

# 3. Model Definition: Regularized L2 Logistic Regression (Credit Risk Benchmark)
# In R: logistic_reg(penalty = 0.01, mixture = 0) %>% set_engine("glmnet")
baseline_model = LogisticRegression(
    penalty='l2',
    C=1.0,               # Inverse regularization strength (smaller = stronger penalty)
    max_iter=1000,       # Guarantees convergence
    random_state=42
)

# Complete Pipeline
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', baseline_model)
])

# 4. Execute Cross-Validation with MLflow Parent Run
with mlflow.start_run(run_name="baseline_logistic_regression_5fold"):
    
    # Log configuration parameters
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "n_splits": 5,
        "features_count": len(candidate_features)
    })
    
    print("Beginning 5-Fold Stratified Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        # Slice fold data
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit on 4 folds (Preprocessor learns scaling on X_tr ONLY - Zero Data Leakage)
        baseline_pipeline.fit(X_tr, y_tr)
        
        # Predict probabilities on the holdout fold
        val_probs = baseline_pipeline.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        
        # Calculate fold AUC
        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc)
        
        # Accumulate fold predictions on the competition test set
        test_preds_folds += baseline_pipeline.predict_proba(X_test)[:, 1] / skf.n_splits
        
        print(f"  Fold {fold} ROC-AUC: {fold_auc:.4f}")
    
    # 5. Overall Out-Of-Fold Evaluation
    overall_oof_auc = roc_auc_score(y, oof_preds)
    mean_cv_auc = np.mean(fold_aucs)
    std_cv_auc = np.std(fold_aucs)
    
    mlflow.log_metric("oof_roc_auc", overall_oof_auc)
    mlflow.log_metric("mean_cv_auc", mean_cv_auc)
    mlflow.log_metric("std_cv_auc", std_cv_auc)
    
    print("\n" + "="*45)
    print(f"Overall OOF ROC-AUC: {overall_oof_auc:.4f}")
    print(f"Mean Fold ROC-AUC:   {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")
    print("="*45)

# %% [6] Generate Baseline Submission from Fold Ensembling
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_status': test_preds_folds   # Average probability across all 5 folds
})

submission.to_csv('data/submission_baseline_logreg.csv', index=False)
print(f"\nSaved submission to data/submission_baseline_logreg.csv (Shape: {submission.shape})")

# %% [7] Challenger Model: 5-Fold Stratified CV with XGBoost
from xgboost import XGBClassifier

# 1. Pipeline Adaptation
# Note: Tree models are scale-invariant, so StandardScaler is technically unnecessary,
# but keeping OneHotEncoder ensures categorical text is numerical.
# Passing preprocessor ensures an identical feature pipeline structure.
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="auc",
    n_jobs=-1            # Utilizes all CPU cores in parallel
)

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# 2. Tracking Setup
oof_preds_xgb = np.zeros(len(train_df))
test_preds_xgb_folds = np.zeros(len(test_df))
fold_aucs_xgb = []

with mlflow.start_run(run_name="challenger_xgboost_5fold"):
    
    # Log hyperparameters
    mlflow.log_params({
        "model_type": "XGBClassifier",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_splits": 5
    })
    
    print("\nBeginning 5-Fold XGBoost Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit fold
        xgb_pipeline.fit(X_tr, y_tr)
        
        # Holdout fold probabilities
        val_probs = xgb_pipeline.predict_proba(X_va)[:, 1]
        oof_preds_xgb[val_idx] = val_probs
        
        # Metric
        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs_xgb.append(fold_auc)
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc)
        
        # Test fold blend
        test_preds_xgb_folds += xgb_pipeline.predict_proba(X_test)[:, 1] / skf.n_splits
        
        print(f"  Fold {fold} ROC-AUC: {fold_auc:.4f}")
        
    overall_oof_auc_xgb = roc_auc_score(y, oof_preds_xgb)
    mean_cv_auc_xgb = np.mean(fold_aucs_xgb)
    std_cv_auc_xgb = np.std(fold_aucs_xgb)
    
    mlflow.log_metric("oof_roc_auc", overall_oof_auc_xgb)
    mlflow.log_metric("mean_cv_auc", mean_cv_auc_xgb)
    mlflow.log_metric("std_cv_auc", std_cv_auc_xgb)
    
    print("\n" + "="*45)
    print(f"XGBoost OOF ROC-AUC: {overall_oof_auc_xgb:.4f}")
    print(f"Mean Fold ROC-AUC:   {mean_cv_auc_xgb:.4f} (+/- {std_cv_auc_xgb:.4f})")
    print(f"Baseline Delta:      {(overall_oof_auc_xgb - overall_oof_auc):+.4f}")
    print("="*45)

# %% [8] Generate XGBoost Submission File
submission_xgb = pd.DataFrame({
    'id': test_df['id'],
    'loan_status': test_preds_xgb_folds
})

submission_xgb.to_csv('data/submission_challenger_xgb.csv', index=False)
print("\nSaved XGBoost predictions to data/submission_challenger_xgb.csv")
# %% [9] Model Understanding: Feature Importance (Gain)
import matplotlib.pyplot as plt

# 1. Extract the fitted preprocessor and model from our pipeline
fitted_preprocessor = xgb_pipeline.named_steps['preprocessor']
fitted_xgb = xgb_pipeline.named_steps['classifier']

# 2. Extract feature names after One-Hot Encoding
# get_feature_names_out() cleanly maps all encoded and scaled column headers
feature_names = fitted_preprocessor.get_feature_names_out()

# Clean up prefixes ('num__', 'cat__') for cleaner chart readability
clean_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

# 3. Extract Feature Importances (Default in XGBoost is 'gain' - total reduction in log-loss)
importances = fitted_xgb.feature_importances_

# 4. Create a readable pandas Series and take top 12
feat_imp = pd.Series(importances, index=clean_names).sort_values(ascending=True)
top_12 = feat_imp.tail(12)

# 5. Plot in Interactive Window
plt.figure(figsize=(9, 6))
top_12.plot(kind='barh', color='#1f77b4', edgecolor='black', alpha=0.85)
plt.title("XGBoost Top 12 Feature Importances (Gain)", fontsize=14, weight='bold')
plt.xlabel("Relative Importance (Gain)", fontsize=11)
plt.ylabel("Features", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print exact values
print("\nTop 12 Most Influential Features:")
print(top_12.sort_values(ascending=False).round(4).to_string())
# %% [10] Model Understanding: SHAP Beeswarm (Directional Impact)
import shap
import matplotlib.pyplot as plt

# 1. Take a representative sample (2,500 rows) for rapid computation
X_sample = X.sample(n=2500, random_state=42)

# 2. Transform the sample using our fitted ColumnTransformer
# Convert back to a DataFrame with our clean feature names so SHAP labels them nicely
X_sample_transformed = fitted_preprocessor.transform(X_sample)
X_sample_df = pd.DataFrame(X_sample_transformed, columns=clean_names)

# 3. Build the TreeExplainer
# TreeExplainer is optimized in C++ directly for XGBoost tree structures
explainer = shap.TreeExplainer(fitted_xgb)
shap_values = explainer(X_sample_df)

# 4. Generate the Beeswarm Plot
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("SHAP Beeswarm Plot: Feature Impact on Loan Default Risk", fontsize=13, weight='bold', pad=15)
plt.xlabel("SHAP Value (Impact on Model Log-Odds: >0 increases default risk, <0 decreases)", fontsize=10)
plt.tight_layout()
plt.show()
# %% [11] Feature Engineering Pipeline
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering to credit applicant data.
    Direct equivalent of dplyr::mutate() or recipes::step_mutate().
    """
    df = df.copy()
    
    # 1. Ordinal Grade Mapping (A=1, B=2, ..., G=7)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['loan_grade_num'] = df['loan_grade'].map(grade_map)
    
    # 2. Binary Flag for Prior Default (0 or 1)
    df['default_flag_num'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
    
    # 3. Bureau History & Employment relative to Age
    # (Adding small epsilon to avoid divide-by-zero)
    df['cred_hist_age_ratio'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1e-5)
    df['emp_age_ratio'] = df['person_emp_length'] / (df['person_age'] + 1e-5)
    
    # 4. Cash Cushion / Disposable Income Proxy
    df['net_income_after_loan'] = df['person_income'] - df['loan_amnt']
    
    # 5. Compound Risk Flag: High DTI + Renter
    df['high_dti_renter'] = (
        (df['loan_percent_income'] > 0.35) & 
        (df['person_home_ownership'] == 'RENT')
    ).astype(int)
    
    return df

# Apply transformations to train and test
print("Engineering features...")
train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

# Define updated feature lists
# Notice: 'loan_grade' and 'cb_person_default_on_file' are now represented numerically!
categorical_fe = ['person_home_ownership', 'loan_intent']

numeric_fe = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'loan_grade_num', 'default_flag_num', 'cred_hist_age_ratio', 
    'emp_age_ratio', 'net_income_after_loan', 'high_dti_renter'
]

# Update candidate feature matrix
X_fe = train_fe[numeric_fe + categorical_fe]
X_test_fe = test_fe[numeric_fe + categorical_fe]

print(f"Original Feature Count: {len(candidate_features)}")
print(f"Engineered Feature Count: {X_fe.shape[1]}")
print(f"Numeric: {len(numeric_fe)}, Categorical: {len(categorical_fe)}")
# %% [12] Train Challenger with Engineered Features
# 1. Update Preprocessor for new column lists
preprocessor_fe = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('scaler', StandardScaler())]), numeric_fe),
        ('cat', Pipeline([('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), categorical_fe)
    ]
)

xgb_fe_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_fe),
    ('classifier', xgb_model)  # Same exact hyperparameters as before
])

# 2. Setup Tracking
oof_preds_fe = np.zeros(len(train_df))
test_preds_fe_folds = np.zeros(len(test_df))
fold_aucs_fe = []

with mlflow.start_run(run_name="challenger_xgb_feature_engineering_5fold"):
    
    mlflow.log_params({
        "model_type": "XGBClassifier",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "features_count": X_fe.shape[1],
        "feature_engineering": "ordinal_grades_ratios_cushion"
    })
    
    print("\nBeginning 5-Fold XGBoost Training with Feature Engineering...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
        X_tr, y_tr = X_fe.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_fe.iloc[val_idx], y.iloc[val_idx]
        
        xgb_fe_pipeline.fit(X_tr, y_tr)
        
        val_probs = xgb_fe_pipeline.predict_proba(X_va)[:, 1]
        oof_preds_fe[val_idx] = val_probs
        
        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs_fe.append(fold_auc)
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc)
        
        test_preds_fe_folds += xgb_fe_pipeline.predict_proba(X_test_fe)[:, 1] / skf.n_splits
        print(f"  Fold {fold} ROC-AUC: {fold_auc:.4f}")
        
    overall_oof_auc_fe = roc_auc_score(y, oof_preds_fe)
    mean_cv_auc_fe = np.mean(fold_aucs_fe)
    std_cv_auc_fe = np.std(fold_aucs_fe)
    
    mlflow.log_metric("oof_roc_auc", overall_oof_auc_fe)
    mlflow.log_metric("mean_cv_auc", mean_cv_auc_fe)
    mlflow.log_metric("std_cv_auc", std_cv_auc_fe)
    
    print("\n" + "="*50)
    print(f"Previous XGBoost OOF ROC-AUC:    {overall_oof_auc_xgb:.4f}")
    print(f"New Engineered XGB OOF ROC-AUC:  {overall_oof_auc_fe:.4f}")
    print(f"Feature Engineering Delta:       {(overall_oof_auc_fe - overall_oof_auc_xgb):+.4f}")
    print("="*50)

# %% [13] Generate Submission with Engineered Features
submission_fe = pd.DataFrame({
    'id': test_df['id'],
    'loan_status': test_preds_fe_folds
})
submission_fe.to_csv('data/submission_xgb_fe.csv', index=False)
print("Saved predictions to data/submission_xgb_fe.csv")
# %%
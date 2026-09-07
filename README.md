Loan Default Risk: Machine Learning Pipeline & Inference ServiceAn end-to-end machine learning system designed to assess credit applicant default risk. The project features automated preprocessing, Bayesian hyperparameter optimization, interpretability diagnostics via SHAP, cross-validated fold ensembling, and a real-time REST API for low-latency scoring.Architecture OverviewPlaintext├── data/                    # Datasets and submission outputs (ignored by git)
├── models/                  # Serialized model binaries (.joblib) and preprocessors
├── reports/                 # Interpretability figures and SHAP beeswarm visualizations
├── src/
│   ├── baseline_cv.py       # Data audit, baseline Logistic Regression & initial XGBoost CV
│   ├── tune_xgb.py          # Optuna Bayesian hyperparameter search with nested MLflow runs
│   ├── train_tuned_xgb.py   # Final 5-fold CV using optimal XGBoost parameters
│   ├── train_lgbm.py        # 5-fold CV LightGBM utilizing native categorical encoding
│   ├── blend_models.py      # Grid-search optimization for champion ensemble weights
│   ├── explain_lgbm.py      # Global SHAP feature attribution computation & export
│   ├── predict.py           # Configurable batch scoring CLI script
│   └── api.py               # FastAPI production inference server with Pydantic schemas
├── mlflow.db                # SQLite metadata tracking store
├── requirements.txt         # Project dependencies
└── README.md
Validation Strategy & Model BenchmarksModels were evaluated using 5-Fold Stratified Cross-Validation, ensuring identical class proportions across all splits without target leakage. The primary evaluation metric is ROC-AUC.  Model StageStrategy / DescriptionOut-Of-Fold ROC-AUCBaselineL2 Regularized Logistic Regression (C=1.0)  ~0.865  Challenger 1Baseline XGBoost (Default Trees, Depth 5)  ~0.948  Feature EngineeredDomain Ratios, Disposable Income Proxy, Risk Flags  ~0.952  Tuned ChallengerOptuna Bayesian Optimized XGBoost (max_depth=7, lr=0.054)  0.95714  Native LightGBMLightGBM with Native Histogram Categorical Binning  0.95803  Champion EnsembleWeighted Blend (0.68 LightGBM + 0.32 XGBoost)  0.95844  Feature Engineering HighlightsDomain-specific risk features engineered in baseline_cv.py:  Credit History to Age Ratio (cred_hist_age_ratio): Depth of credit maturity relative to overall applicant life experience.  Employment Stability Ratio (emp_age_ratio): Proportion of working life spent in current employment.  Disposable Income Proxy (net_income_after_loan): Residual financial buffer calculated as person_income - loan_amnt.  Compound Risk Flag (high_dti_renter): Indicator flagging high debt burden (loan_percent_income > 0.35) paired with rental occupancy.  Ordinal Grade Mapping (loan_grade_num): Monotonic integer mapping (A=1 through G=7) replacing standard nominal dummy columns.  Experiment Tracking (MLflow)All training runs, hyperparameters, fold metrics, diagnostic visualizations, and model binaries are logged to a local SQLite-backed MLflow registry:  Backend Store: sqlite:///mlflow.db  Tracked Parameters: Tree depth, learning rate, regularization (alpha/lambda), subsampling rates, and split parameters.  Tracked Artifacts: Out-of-fold probability vectors, serialized .joblib estimators, SHAP plots, and submission CSVs.  To launch the tracking interface:Bashmlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to review runs and parameter charts.Model Interpretability (SHAP)Global feature attribution was computed using tree-specific SHAP (TreeExplainer):  Top Risk Drivers: loan_percent_income and loan_int_rate exhibit the strongest positive association with default probability.  Mitigating Factors: Higher annual income and longer credit bureau history significantly reduce predicted risk.  Diagnostic beeswarm outputs are generated via src/explain_lgbm.py and exported directly into reports/.  Serving & Inference1. Real-Time REST API (FastAPI)The application serves the champion 10-fold ensemble via an asynchronous FastAPI service:Lifespan Caching: Pre-loads all 5 LightGBM fold models, 5 XGBoost fold models, and the fitted ColumnTransformer into RAM on startup, achieving single-digit millisecond response times.Strict Validation: Request validation powered by Pydantic v2 schemas rejects malformed or missing applicant data before scoring.Launch the API server:Bashpython -m uvicorn src.api:app --reload --port 8000
Interactive OpenAPI / Swagger documentation is available at:Plaintexthttp://127.0.0.1:8000/docs
Sample Request Payload (POST /predict):JSON{
  "person_age": 26,
  "person_income": 58000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000.0,
  "loan_int_rate": 11.25,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}
Sample Response Body:JSON{
  "loan_default_probability": 0.1142,
  "model_breakdown": {
    "lightgbm_mean": 0.1085,
    "xgboost_mean": 0.1263
  },
  "decision": "APPROVE"
}
2. Batch Scoring CLIScore static CSV batches directly from the terminal without starting the web server:Bash# Score default test file
python src/predict.py

# Score custom batch file with custom output destination
python src/predict.py --input data/new_applicants.csv --output data/new_scored.csv
Installation & SetupClone the repository:Bashgit clone https://github.com/mgredlics/Loan_Approval.git
cd Loan_Approval
Set up a virtual environment:Bashpython -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
Install required packages:Bashpip install -r requirements.txt
Train and reproduce pipeline:Bashpython src/train_tuned_xgb.py
python src/train_lgbm.py
python src/blend_models.py
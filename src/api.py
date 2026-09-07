#python -m uvicorn src.api:app --reload --port 8000
#http://127.0.0.1:8000/docs

import contextlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Request Schema (Pydantic)
# In R Plumber, you manually parse req$postBody.
# In FastAPI, Pydantic type hints validate and coerce incoming JSON payloads.
# ---------------------------------------------------------------------------
class LoanApplicant(BaseModel):
  person_age: int = Field(..., examples=[26])
  person_income: float = Field(..., examples=[58000.0])
  person_home_ownership: str = Field(..., examples=["RENT"])
  person_emp_length: float = Field(..., examples=[3.0])
  loan_intent: str = Field(..., examples=["PERSONAL"])
  loan_grade: str = Field(..., examples=["B"])
  loan_amnt: float = Field(..., examples=[10000.0])
  loan_int_rate: float = Field(..., examples=[11.25])
  loan_percent_income: float = Field(..., examples=[0.17])
  cb_person_default_on_file: str = Field(..., examples=["N"])
  cb_person_cred_hist_length: int = Field(..., examples=[4])


# Container to hold preloaded models in memory
model_artifacts = {"lgbm_models": [], "xgb_models": [], "xgb_preprocessor": None}


# ---------------------------------------------------------------------------
# 2. Application Lifespan (Load Models Once on Startup)
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
  print("Loading models and preprocessor into server memory...")
  models_dir = PROJECT_ROOT / "models"

  # Load 5 LightGBM folds
  model_artifacts["lgbm_models"] = [
      joblib.load(models_dir / f"lgbm_fold_{i}.joblib") for i in range(1, 6)
  ]

  # Load XGBoost preprocessor and 5 folds
  model_artifacts["xgb_preprocessor"] = joblib.load(
      models_dir / "xgb_preprocessor.joblib"
  )
  model_artifacts["xgb_models"] = [
      joblib.load(models_dir / f"xgb_fold_{i}.joblib") for i in range(1, 6)
  ]

  print("All 10 fold models and preprocessor successfully loaded.")
  yield
  model_artifacts.clear()


app = FastAPI(
    title="Loan Approval Inference Service",
    description="Real-time default risk prediction using champion LightGBM/XGBoost ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 3. Healthcheck Endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
def healthcheck():
  return {
      "status": "healthy",
      "models_loaded": len(model_artifacts["lgbm_models"]) == 5,
  }

@app.get("/version")
def get_version():
  return {
      "service": "Loan Approval Inference Service",
      "api_version": "1.0.0",
      "champion_ensemble_weights": {"lightgbm": 0.68, "xgboost": 0.32},
  }
# ---------------------------------------------------------------------------
# 4. Prediction Scoring Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
def predict_default(applicant: LoanApplicant):
  try:
    # Convert incoming validated record to a single-row DataFrame
    input_data = pd.DataFrame([applicant.model_dump()])

    # 1. LightGBM Inference (Native Categorical Dtypes)
    X_lgb = input_data.copy()
    cat_cols = X_lgb.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
      X_lgb[col] = X_lgb[col].astype("category")

    lgb_probs = [
        m.predict_proba(X_lgb)[0, 1] for m in model_artifacts["lgbm_models"]
    ]
    mean_lgb_prob = float(np.mean(lgb_probs))

    # 2. XGBoost Inference (ColumnTransformer Transformation)
    X_xgb_proc = model_artifacts["xgb_preprocessor"].transform(input_data)
    xgb_probs = [
        m.predict_proba(X_xgb_proc)[0, 1] for m in model_artifacts["xgb_models"]
    ]
    mean_xgb_prob = float(np.mean(xgb_probs))

    # 3. Apply Champion Ensemble Weights (0.68 LGBM + 0.32 XGB)
    ensemble_prob = float((0.68 * mean_lgb_prob) + (0.32 * mean_xgb_prob))

    # Decision rule heuristic: Approve if default probability < 0.25
    decision = "DECLINE" if ensemble_prob >= 0.25 else "APPROVE"

    return {
        "loan_default_probability": round(ensemble_prob, 4),
        "model_breakdown": {
            "lightgbm_mean": round(mean_lgb_prob, 4),
            "xgboost_mean": round(mean_xgb_prob, 4),
        },
        "decision": decision,
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. Direct Execution Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import uvicorn

  uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
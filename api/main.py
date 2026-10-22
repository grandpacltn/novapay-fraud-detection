"""
NovaPay Fraud Detection API

Run from the project root (the Fraud folder):
    uvicorn api.main:app --reload

Swagger docs:
    http://127.0.0.1:8000/docs
"""
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load the saved model, its feature list and the SHAP explainer ONCE at startup
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model = joblib.load(MODELS_DIR / "fraud_model.joblib")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
explainer = shap.TreeExplainer(model)
THRESHOLD = 0.5  # chosen during threshold tuning

app = FastAPI(
    title="NovaPay Fraud Detection API",
    description=(
        "Scores a NovaPay transaction and returns the **fraud probability**, "
        "a **risk level**, and the **top reasons** behind the decision (using SHAP).\n\n"
        "Model: tuned Random Forest (PR-AUC 0.88, precision 0.948, recall 0.819)."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Input: what a transaction looks like (Swagger builds its form from this)
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    timestamp: datetime = Field(..., description="Transaction time in UTC")
    home_country: Literal["US", "UK", "CA"]
    source_currency: Literal["USD", "GBP", "CAD"]
    dest_currency: Literal["USD", "GBP", "CAD", "EUR", "NGN", "INR", "CNY", "PHP", "MXN"]
    channel: Literal["mobile", "web", "atm"]
    amount_src: float = Field(..., gt=0, description="Amount in the source currency")
    amount_usd: float = Field(..., gt=0, description="Amount converted to USD")
    fee: float = Field(..., ge=0)
    exchange_rate_src_to_dest: float = Field(..., gt=0)
    new_device: bool
    location_mismatch: bool = Field(..., description="IP country differs from home country")
    ip_country: Literal["US", "UK", "CA", "unknown"]
    ip_risk_score: float = Field(..., ge=0, le=1)
    kyc_tier: Literal["standard", "enhanced", "low", "unknown"]
    account_age_days: int = Field(..., ge=0)
    device_trust_score: float = Field(..., ge=0, le=1)
    chargeback_history_count: int = Field(..., ge=0)
    txn_velocity_1h: int = Field(..., ge=0, description="Transactions in the last hour")
    txn_velocity_24h: int = Field(..., ge=0, description="Transactions in the last 24 hours")
    corridor_risk: float = Field(..., ge=0)
    # Behaviour features: in production these would come from the customer's history
    cust_txn_count: int = Field(1, ge=1, description="Customer's number of transactions on record")
    cust_avg_amount_usd: float | None = Field(None, gt=0, description="Customer's usual amount in USD")
    device_cust_count: int = Field(1, ge=1, description="Customers who have used this device")
    ip_txn_count: int = Field(1, ge=1, description="Transactions from this IP address")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "timestamp": "2026-10-22T05:30:00Z",
                "home_country": "US",
                "source_currency": "USD",
                "dest_currency": "NGN",
                "channel": "web",
                "amount_src": 1500.0,
                "amount_usd": 1500.0,
                "fee": 25.0,
                "exchange_rate_src_to_dest": 1388.89,
                "new_device": True,
                "location_mismatch": True,
                "ip_country": "UK",
                "ip_risk_score": 0.688,
                "kyc_tier": "low",
                "account_age_days": 63,
                "device_trust_score": 0.215,
                "chargeback_history_count": 1,
                "txn_velocity_1h": 8,
                "txn_velocity_24h": 11,
                "corridor_risk": 0.25,
                "cust_txn_count": 1,
                "cust_avg_amount_usd": 1500.0,
                "device_cust_count": 1,
                "ip_txn_count": 1,
            }]
        }
    }


# ---------------------------------------------------------------------------
# Output: what the API sends back
# ---------------------------------------------------------------------------
class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    top_reasons: list[str]


# ---------------------------------------------------------------------------
# Rebuild exactly the same features the model was trained on
# ---------------------------------------------------------------------------
def build_features(t: Transaction) -> pd.DataFrame:
    d = t.model_dump()
    hour = d.pop("timestamp").hour
    cust_avg = d.pop("cust_avg_amount_usd") or d["amount_usd"]

    d["hour"] = hour
    d["is_night"] = int(3 <= hour <= 8)
    d["new_account"] = int(d["account_age_days"] < 50)
    d["risky_corridor"] = int(d["dest_currency"] in ("NGN", "MXN"))
    d["fee_pct"] = d["fee"] / d["amount_usd"]
    d["velocity_ratio"] = d["txn_velocity_1h"] / (d["txn_velocity_24h"] + 1)
    d["amount_vs_cust_avg"] = d["amount_usd"] / cust_avg

    row = pd.get_dummies(
        pd.DataFrame([d]),
        columns=["home_country", "source_currency", "dest_currency",
                 "channel", "ip_country", "kyc_tier"],
    )
    # Same columns, same order as training; missing dummy columns become 0
    return row.reindex(columns=feature_columns, fill_value=0).astype(float)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def home():
    return {"message": "NovaPay Fraud Detection API. Go to /docs to try it."}


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "model_loaded": True, "features": len(feature_columns)}


@app.post("/predict", response_model=Prediction, tags=["Fraud Detection"])
def predict(transaction: Transaction):
    """Score one transaction and explain the decision."""
    X = build_features(transaction)
    proba = float(model.predict_proba(X)[0, 1])

    # SHAP: how much each feature pushed THIS transaction towards fraud
    contributions = explainer(X).values[0, :, 1]
    top = sorted(zip(feature_columns, X.iloc[0], contributions),
                 key=lambda item: item[2], reverse=True)[:3]
    reasons = [f"{name} = {value:g} (+{impact:.2f})" for name, value, impact in top if impact > 0]

    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.3 else "low"
    return Prediction(
        fraud_probability=round(proba, 3),
        is_fraud=proba >= THRESHOLD,
        risk_level=risk,
        top_reasons=reasons,
    )

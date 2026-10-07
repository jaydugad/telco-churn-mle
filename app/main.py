from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from src.predict import predict

STATIC_DIR = Path(__file__).parent / "static"

# Allowed category values (taken from the training data)
YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

SAMPLE = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85",
}

app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Scores a single customer profile and returns the probability that they "
        "will churn. Built on a tuned XGBoost pipeline (test ROC-AUC 0.846, PR-AUC 0.663)."
    ),
    version="1.0.0",
)


class CustomerProfile(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": SAMPLE})

    customerID: Optional[str] = None
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[float, str] = Field(description="Raw data stores this as text")


class PredictionResponse(BaseModel):
    customerID: Optional[str]
    churn_prediction: int
    churn_probability: float


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Simple web form for trying the model in a browser."""
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerProfile):
    """Return the churn prediction and probability for one customer."""
    try:
        return predict(customer.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
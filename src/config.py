from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

# Columns
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
import pandas as pd

from src.config import DATA_PATH, TARGET, ID_COL


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning rules. Used in both training and inference."""
    df = df.copy()

    # TotalCharges is stored as text; blanks become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Blanks only occur for brand-new customers (tenure = 0) who haven't been billed yet
    new_customers = df["tenure"] == 0
    df.loc[new_customers, "TotalCharges"] = df.loc[new_customers, "TotalCharges"].fillna(0)

    # SeniorCitizen is 0/1 but represents a category, not a quantity
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Customer ID carries no predictive information
    df = df.drop(columns=[ID_COL], errors="ignore")

    return df


def load_data():
    """Load the raw CSV and return cleaned features X and binary target y."""
    df = pd.read_csv(DATA_PATH)
    y = (df[TARGET] == "Yes").astype(int)
    X = clean_features(df.drop(columns=[TARGET]))
    return X, y
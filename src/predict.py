import json
import sys
from functools import lru_cache

import joblib
import pandas as pd

from src.config import ID_COL, MODEL_PATH
from src.data import clean_features

THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline once and reuse it across predictions."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict(payload: dict) -> dict:
    """Predict churn for a single raw customer profile."""
    customer_id = payload.get(ID_COL)
    X = clean_features(pd.DataFrame([payload]))  # same cleaning as training
    probability = float(load_model().predict_proba(X)[0, 1])
    return {
        "customerID": customer_id,
        "churn_prediction": int(probability >= THRESHOLD),
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    with open(path) as f:
        print(json.dumps(predict(json.load(f)), indent=2))
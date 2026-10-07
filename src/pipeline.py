from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing that is fitted on training data only (prevents leakage)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


def build_pipeline(model) -> Pipeline:
    """Combine preprocessing and any sklearn-compatible model into one pipeline."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
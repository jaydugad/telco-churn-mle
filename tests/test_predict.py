import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from src.predict import predict

SAMPLE = json.loads((Path(__file__).parent.parent / "sample_input.json").read_text())
client = TestClient(app)


def test_predict_function_returns_valid_output():
    result = predict(SAMPLE)
    assert result["churn_prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_api_predict_endpoint():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert set(response.json()) == {"customerID", "churn_prediction", "churn_probability"}


def test_api_rejects_invalid_payload():
    bad = {**SAMPLE, "tenure": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_api_rejects_unknown_category():
    bad = {**SAMPLE, "Contract": "Weekly"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
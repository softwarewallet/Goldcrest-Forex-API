from fastapi.testclient import TestClient
from app.config import get_settings
from app.main import app

def test_model_comparison_endpoint_requires_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/research/model-comparison")
    assert response.status_code in {401, 503}

def test_model_comparison_endpoint_returns_research_only_contract():
    headers = {"X-API-Key": get_settings().api_key.get_secret_value()}
    with TestClient(app) as client:
        response = client.get("/api/v1/research/model-comparison", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["accuracy_metrics_published"] is False
    assert isinstance(body["models"], dict)
    assert isinstance(body["comparisons"], list)

def test_model_comparison_rejects_invalid_sample_floor():
    headers = {"X-API-Key": get_settings().api_key.get_secret_value()}
    with TestClient(app) as client:
        response = client.get("/api/v1/research/model-comparison?minimum_resolved_samples=0", headers=headers)
    assert response.status_code == 422

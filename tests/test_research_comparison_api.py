from fastapi.testclient import TestClient
from app.config import get_settings
from app.main import app


def test_model_comparison_is_protected():
    with TestClient(app) as client:
        response = client.get("/api/v1/research/model-comparison")
    assert response.status_code in {401, 403, 503}


def test_model_comparison_returns_stratified_research_contract():
    key = get_settings().api_key.get_secret_value()
    with TestClient(app) as client:
        response = client.get("/api/v1/research/model-comparison?minimum_resolved_samples=5", headers={"X-API-Key": key})
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["minimum_resolved_samples"] == 5
    assert body["accuracy_metrics_published"] is False
    assert isinstance(body["strata"], list)
    assert "AMBIGUOUS" in body["excluded_outcomes"]
    for group in body["strata"]:
        assert group["symbol"] and group["timeframe"]
        assert isinstance(group["models"], dict)
        assert isinstance(group["comparisons"], list)


def test_model_comparison_rejects_invalid_query_parameters():
    key = get_settings().api_key.get_secret_value()
    with TestClient(app) as client:
        low = client.get("/api/v1/research/model-comparison?minimum_resolved_samples=0", headers={"X-API-Key": key})
        high = client.get("/api/v1/research/model-comparison?credible_level=1", headers={"X-API-Key": key})
    assert low.status_code == 422
    assert high.status_code == 422

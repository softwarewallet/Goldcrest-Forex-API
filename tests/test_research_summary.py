from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_research_summary_requires_api_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/research/summary")

    assert response.status_code in {401, 503}


def test_research_summary_returns_consistent_aggregate_counts():
    headers = {"X-API-Key": get_settings().api_key.get_secret_value()}
    with TestClient(app) as client:
        response = client.get("/api/v1/research/summary", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert isinstance(body["predictions_total"], int)
    assert isinstance(body["outcomes_recorded"], int)
    assert isinstance(body["pending_outcomes"], int)
    assert body["predictions_total"] >= body["outcomes_recorded"]
    assert body["pending_outcomes"] == body["predictions_total"] - body["outcomes_recorded"]
    assert body["accuracy_metrics_published"] is False
    assert "not directional accuracy" in body["note"]
    assert isinstance(body["decisions"], list)
    assert isinstance(body["outcomes"], list)
    assert isinstance(body["models"], list)

    if body["predictions_total"]:
        assert 0 <= body["outcome_coverage"] <= 1
    else:
        assert body["outcome_coverage"] is None


def test_performance_returns_counts_without_publishing_accuracy():
    headers = {"X-API-Key": get_settings().api_key.get_secret_value()}
    with TestClient(app) as client:
        response = client.get("/api/v1/performance", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predictions_total"], int)
    assert isinstance(body["outcomes_recorded"], int)
    assert "not published" in body["note"]

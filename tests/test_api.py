from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def api_key():
    return get_settings().api_key.get_secret_value()


def market_payload(request_id="test-req-default"):
    now = datetime.now(timezone.utc)
    candles = []
    price = 1.1
    for i in range(40):
        close = price + 0.0001
        candles.append({
            "timestamp": (now - timedelta(minutes=5 * (39 - i))).isoformat(),
            "open": price,
            "high": close + 0.0002,
            "low": price - 0.0002,
            "close": close,
            "volume": 100,
        })
        price = close
    return {
        "request_id": request_id,
        "symbol": "EUR/USD",
        "timeframe": "M5",
        "market_data": {
            "bid": 1.1,
            "ask": 1.1001,
            "snapshot_at": now.isoformat(),
            "candles": candles,
        },
    }


def test_health_route_does_not_depend_on_external_model_server(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_list_models(self):
        return [{"id": "test-model"}]

    monkeypatch.setattr(LlamaClient, "list_models", fake_list_models)
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert {"status", "inference_status", "database_status", "api_key_configured"} <= body.keys()
    assert body["database_status"] == "ok"
    assert body["inference_status"] == "ok"
    assert body["api_key_configured"] is get_settings().api_key_configured


def test_protected_route_without_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/models")
    assert response.status_code == 401


def test_invalid_ohlc_rejected():
    body = market_payload("test-invalid-ohlc")
    body["market_data"]["candles"][2]["high"] = 0.1
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            headers={"X-API-Key": api_key()},
            json=body,
        )
    assert response.status_code == 422


def test_stale_snapshot_rejected(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload("test-stale-snapshot")
    body["market_data"]["snapshot_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=20)
    ).isoformat()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            headers={"X-API-Key": api_key()},
            json=body,
        )
    assert response.status_code == 422


def test_llm_buy_suggestion_is_not_actionable_before_validation(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", (
            '{"decision":"BUY","market_regime":"TRENDING_UP",'
            '"reason_codes":["TREND_CONFIRMED"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload("test-buy-remains-no-trade")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            headers={"X-API-Key": api_key()},
            json=body,
        )

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "NO_TRADE"
    assert result["probability"] is None
    assert "BASELINE_ONLY_NO_VALIDATED_EDGE" in result["reason_codes"]


def test_inference_failure_fails_safe(monkeypatch):
    from app.llama_client import InferenceUnavailable, LlamaClient

    async def failed_analyze(self, payload):
        raise InferenceUnavailable("simulated test outage")

    monkeypatch.setattr(LlamaClient, "analyze", failed_analyze)
    body = market_payload("test-inference-failure")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            headers={"X-API-Key": api_key()},
            json=body,
        )

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "NO_TRADE"
    assert result["probability"] is None


def test_reusing_request_id_with_different_payload_returns_conflict(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload("test-id-payload-conflict")
    headers = {"X-API-Key": api_key()}
    with TestClient(app) as client:
        first = client.post("/api/v1/predictions", headers=headers, json=body)
        assert first.status_code == 200
        changed = {**body, "market_data": {**body["market_data"], "bid": 1.0999}}
        second = client.post("/api/v1/predictions", headers=headers, json=changed)

    assert second.status_code == 409
    assert second.json()["detail"] == "request_id already used with a different payload"


def test_same_request_id_and_same_payload_returns_existing_prediction(monkeypatch):
    from app.llama_client import LlamaClient

    calls = {"count": 0}

    async def fake_analyze(self, payload):
        calls["count"] += 1
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload("test-idempotent-replay")
    headers = {"X-API-Key": api_key()}
    with TestClient(app) as client:
        first = client.post("/api/v1/predictions", headers=headers, json=body)
        second = client.post("/api/v1/predictions", headers=headers, json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["prediction_id"] == first.json()["prediction_id"]
    assert calls["count"] == 1


def test_persisted_prediction_timestamps_are_utc_aware(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload("test-persisted-utc-timestamps")
    headers = {"X-API-Key": api_key()}
    with TestClient(app) as client:
        created = client.post("/api/v1/predictions", headers=headers, json=body)
        prediction_id = created.json()["prediction_id"]
        fetched = client.get(f"/api/v1/predictions/{prediction_id}", headers=headers)

    assert created.status_code == 200
    assert fetched.status_code == 200
    generated_at = datetime.fromisoformat(fetched.json()["generated_at"])
    expires_at = datetime.fromisoformat(fetched.json()["expires_at"])
    assert generated_at.utcoffset() == timedelta(0)
    assert expires_at.utcoffset() == timedelta(0)


def test_performance_returns_database_aggregate_counts():
    with TestClient(app) as client:
        response = client.get("/api/v1/performance", headers={"X-API-Key": api_key()})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predictions_total"], int)
    assert body["predictions_total"] >= 0
    assert isinstance(body["outcomes_recorded"], int)
    assert body["outcomes_recorded"] >= 0
    assert "not published" in body["note"]


def test_outcome_details_payload_is_size_limited():
    import pytest
    from pydantic import ValidationError
    from app.schemas import OutcomeRequest

    with pytest.raises(ValidationError, match="8 KiB"):
        OutcomeRequest(
            prediction_id="prediction-test",
            outcome_label="UNKNOWN",
            observed_at=datetime.now(timezone.utc),
            details={"payload": "x" * 9000},
        )


def test_inference_readiness_runs_and_validates_completion(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        assert payload["readiness_probe"] is True
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    with TestClient(app) as client:
        response = client.get("/api/v1/readiness/inference", headers={"X-API-Key": api_key()})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_id": "test-model", "completion_valid": True
    }


def test_inference_readiness_fails_closed_when_model_unavailable(monkeypatch):
    from app.llama_client import InferenceUnavailable, LlamaClient

    async def failed_analyze(self, payload):
        raise InferenceUnavailable("simulated outage")

    monkeypatch.setattr(LlamaClient, "analyze", failed_analyze)
    with TestClient(app) as client:
        response = client.get("/api/v1/readiness/inference", headers={"X-API-Key": api_key()})

    assert response.status_code == 503
    assert response.json()["detail"] == "Inference completion or response validation failed"


def test_prediction_history_requires_api_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/predictions?limit=5")
    assert response.status_code == 401


def test_prediction_history_returns_recent_records_with_limit(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "dashboard-test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["DASHBOARD_HISTORY_TEST"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    headers = {"X-API-Key": api_key()}
    body = market_payload("dashboard-history-list-test")
    with TestClient(app) as client:
        created = client.post("/api/v1/predictions", headers=headers, json=body)
        assert created.status_code == 200
        response = client.get("/api/v1/predictions?limit=1", headers=headers)

    assert response.status_code == 200
    rows = response.json()
    assert isinstance(rows, list)
    assert len(rows) <= 1
    assert rows[0]["prediction_id"] == created.json()["prediction_id"]
    assert rows[0]["decision"] == "NO_TRADE"


def test_dashboard_cors_allows_configured_origin_and_api_key_header():
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/predictions",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "X-API-Key" in response.headers["access-control-allow-headers"]


def test_dashboard_cors_rejects_unconfigured_origin():
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/predictions",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers

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
    assert {"status", "inference_status", "database_status"} <= body.keys()
    assert body["database_status"] == "ok"
    assert body["inference_status"] == "ok"


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

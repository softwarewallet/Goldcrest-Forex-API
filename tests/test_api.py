from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def api_key():
    return get_settings().api_key.get_secret_value()


def market_payload():
    now = datetime.now(timezone.utc)
    candles = []
    price = 1.1
    for i in range(40):
        close = price + 0.0001
        candles.append({"timestamp": (now - timedelta(minutes=5 * (39 - i))).isoformat(),
                        "open": price, "high": close + 0.0002, "low": price - 0.0002,
                        "close": close, "volume": 100})
        price = close
    return {"request_id": "test-req-001", "symbol": "EUR/USD", "timeframe": "M5",
            "market_data": {"bid": 1.1, "ask": 1.1001, "snapshot_at": now.isoformat(),
                            "candles": candles}}


def test_health_route():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert {"status", "inference_status", "database_status"} <= response.json().keys()


def test_protected_route_without_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/models")
    assert response.status_code in (401, 503)


def test_invalid_ohlc_rejected():
    body = market_payload()
    body["market_data"]["candles"][2]["high"] = 0.1
    with TestClient(app) as client:
        response = client.post("/api/v1/predictions", headers={"X-API-Key": api_key()}, json=body)
    assert response.status_code == 422


def test_stale_snapshot_rejected(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", '{"decision":"NO_TRADE","market_regime":"UNCERTAIN","reason_codes":["INSUFFICIENT_EVIDENCE"]}'

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    body = market_payload()
    body["market_data"]["snapshot_at"] = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    with TestClient(app) as client:
        response = client.post("/api/v1/predictions", headers={"X-API-Key": api_key()}, json=body)
    assert response.status_code == 422

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def api_key():
    return get_settings().api_key.get_secret_value()


def market_payload(request_id):
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


def create_test_prediction(client, monkeypatch, request_id):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", (
            '{"decision":"NO_TRADE","market_regime":"UNCERTAIN",'
            '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        )

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    response = client.post(
        "/api/v1/predictions",
        headers={"X-API-Key": api_key()},
        json=market_payload(request_id),
    )
    assert response.status_code == 200
    return response.json()["prediction_id"]


def test_outcome_can_be_recorded_once_and_duplicate_is_conflict(monkeypatch):
    with TestClient(app) as client:
        prediction_id = create_test_prediction(client, monkeypatch, "outcome-once-test")
        body = {
            "prediction_id": prediction_id,
            "outcome_label": "HORIZON_EXPIRED",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "details": {"source": "automated-test"},
        }
        headers = {"X-API-Key": api_key()}

        first = client.post("/api/v1/outcomes", headers=headers, json=body)
        second = client.post("/api/v1/outcomes", headers=headers, json=body)

    assert first.status_code == 200
    assert first.json() == {"status": "recorded", "prediction_id": prediction_id}
    assert second.status_code == 409
    assert second.json()["detail"] == "Outcome already recorded"


def test_outcome_with_future_observation_time_is_rejected(monkeypatch):
    with TestClient(app) as client:
        prediction_id = create_test_prediction(client, monkeypatch, "outcome-future-test")
        response = client.post(
            "/api/v1/outcomes",
            headers={"X-API-Key": api_key()},
            json={
                "prediction_id": prediction_id,
                "outcome_label": "UNKNOWN",
                "observed_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                "details": {},
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "observed_at cannot be in the future"


def test_outcome_for_unknown_prediction_returns_not_found():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/outcomes",
            headers={"X-API-Key": api_key()},
            json={
                "prediction_id": "missing-prediction-id",
                "outcome_label": "UNKNOWN",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "details": {},
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"

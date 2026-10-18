from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.main import app


def api_key():
    return get_settings().api_key.get_secret_value()


def market_payload(request_id="pipeline-test"):
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


def valid_model_output():
    return (
        '{"decision":"NO_TRADE","market_regime":"TRENDING_UP",'
        '"reason_codes":["INSUFFICIENT_EVIDENCE"]}'
    )


def test_inference_readiness_performs_and_validates_completion(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        assert payload["readiness_probe"] is True
        return "test-model", valid_model_output()

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/readiness/inference",
            headers={"X-API-Key": api_key()},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "test-model",
        "completion_valid": True,
    }


def test_inference_readiness_fails_closed_on_invalid_model_output(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", "This is prose, not the required JSON object."

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/readiness/inference",
            headers={"X-API-Key": api_key()},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Inference completion or response validation failed"


def test_prediction_database_failure_returns_controlled_503_and_rolls_back(monkeypatch):
    from app.llama_client import LlamaClient

    async def fake_analyze(self, payload):
        return "test-model", valid_model_output()

    rollback_calls = {"count": 0}
    original_rollback = Session.rollback

    def fail_commit(self):
        raise OperationalError("INSERT prediction", {}, RuntimeError("simulated database failure"))

    def track_rollback(self):
        rollback_calls["count"] += 1
        return original_rollback(self)

    monkeypatch.setattr(LlamaClient, "analyze", fake_analyze)
    monkeypatch.setattr(Session, "commit", fail_commit)
    monkeypatch.setattr(Session, "rollback", track_rollback)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            headers={"X-API-Key": api_key()},
            json=market_payload("pipeline-db-failure"),
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction persistence failed; retry with the same request_id"
    )
    assert rollback_calls["count"] >= 1

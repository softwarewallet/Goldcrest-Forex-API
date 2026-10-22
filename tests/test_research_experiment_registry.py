from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def _headers():
    return {"X-API-Key": get_settings().api_key.get_secret_value()}


def _payload(run_id=None):
    return {
        "run_id": run_id or f"registry-test-{uuid4()}",
        "dataset_sha256": "a" * 64,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "protocol_version": "walk-forward-v1",
        "manifest": {
            "schema_version": 1,
            "dataset_sha256": "a" * 64,
            "fold_count": 2,
            "actionable": False,
            "automatic_winner_selection": False,
            "test_metrics_used_for_selection": False,
            "experiments": [{
                "experiment_id": "xgb-full",
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "actionable": False,
                "test_metrics_used_for_tuning": False,
                "fold_results": [{"fold_index": 0, "train_rows": 100, "test_rows": 20}],
            }],
        },
    }


def test_experiment_registry_requires_authentication():
    with TestClient(app) as client:
        response = client.get("/api/v1/research/experiments")
    assert response.status_code in {401, 403, 503}


def test_experiment_registry_registers_lists_and_fetches_immutable_run():
    payload = _payload()
    with TestClient(app) as client:
        created = client.post("/api/v1/research/experiments", headers=_headers(), json=payload)
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["run_id"] == payload["run_id"]
        assert body["fingerprint_sha256"]
        assert body["actionable"] is False
        assert body["automatic_winner_selection"] is False

        repeated = client.post("/api/v1/research/experiments", headers=_headers(), json=payload)
        assert repeated.status_code == 200
        assert repeated.json()["fingerprint_sha256"] == body["fingerprint_sha256"]

        listed = client.get("/api/v1/research/experiments?symbol=eurusd&timeframe=m15", headers=_headers())
        assert listed.status_code == 200
        assert any(row["run_id"] == payload["run_id"] for row in listed.json()["runs"])

        fetched = client.get(f"/api/v1/research/experiments/{payload['run_id']}", headers=_headers())
        assert fetched.status_code == 200
        assert fetched.json()["fold_results"][0]["fold_results"][0]["fold_index"] == 0


def test_experiment_registry_rejects_bad_hash_and_actionable_manifest():
    with TestClient(app) as client:
        bad_hash = _payload()
        bad_hash["dataset_sha256"] = "not-a-hash"
        assert client.post("/api/v1/research/experiments", headers=_headers(), json=bad_hash).status_code == 422

        unsafe = _payload()
        unsafe["manifest"]["automatic_winner_selection"] = True
        assert client.post("/api/v1/research/experiments", headers=_headers(), json=unsafe).status_code == 422


def test_experiment_registry_rejects_run_id_reuse_with_changed_content():
    payload = _payload()
    with TestClient(app) as client:
        first = client.post("/api/v1/research/experiments", headers=_headers(), json=payload)
        assert first.status_code == 200, first.text
        changed = {**payload, "manifest": {**payload["manifest"], "protocol_note": "changed"}}
        conflict = client.post("/api/v1/research/experiments", headers=_headers(), json=changed)
        assert conflict.status_code == 409

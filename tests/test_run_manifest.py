from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.run_manifest import ExperimentRegistry, build_run_manifest


def make_bars():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        ForexBar("EUR/USD", "M5", start + timedelta(minutes=5 * i),
                 1.1 + i * 0.0001, 1.101 + i * 0.0001,
                 1.099 + i * 0.0001, 1.1005 + i * 0.0001,
                 1.1004 + i * 0.0001, 1.1006 + i * 0.0001)
        for i in range(5)
    ]


def manifest(bars=None, config=None):
    return build_run_manifest(
        bars or make_bars(),
        code_commit="abc123",
        model_id="qwen-local",
        model_revision="revision-1",
        dataset_id="eurusd-m5-fixture-v1",
        evaluation_config=config or {"horizon_bars": 12, "cost_model": "explicit-v1"},
    )


def test_identical_inputs_produce_identical_run_id():
    first = manifest()
    second = manifest()
    assert first.run_id == second.run_id
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.config_fingerprint == second.config_fingerprint


def test_data_or_config_change_changes_run_identity():
    bars = make_bars()
    changed = list(bars)
    changed[2] = ForexBar(
        changed[2].symbol, changed[2].timeframe, changed[2].timestamp,
        changed[2].open, changed[2].high, changed[2].low, changed[2].close + 0.0001,
        changed[2].bid, changed[2].ask,
    )
    assert manifest(changed).run_id != manifest().run_id
    assert manifest(config={"horizon_bars": 24, "cost_model": "explicit-v1"}).run_id != manifest().run_id


def test_naive_timestamps_and_missing_metadata_fail_closed():
    bars = make_bars()
    bars[0] = ForexBar("EUR/USD", "M5", datetime(2025, 1, 1),
                       1.1, 1.101, 1.099, 1.1005)
    with pytest.raises(ValueError, match="timezone-aware"):
        manifest(bars)
    with pytest.raises(ValueError, match="model_revision"):
        build_run_manifest(make_bars(), code_commit="abc", model_id="qwen",
                           model_revision="", dataset_id="data", evaluation_config={})


def test_registry_idempotency_and_result_persistence(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite")
    run = manifest()
    assert registry.register(run) is True
    assert registry.register(run) is False
    registry.record_result(run.run_id, status="COMPLETED",
                           outcome_summary={"labeled": 42, "ambiguous": 3})
    stored = registry.get(run.run_id)
    assert stored["status"] == "COMPLETED"
    assert stored["outcome_summary"] == {"ambiguous": 3, "labeled": 42}


def test_registry_rejects_unknown_run_and_bad_status(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite")
    with pytest.raises(KeyError, match="unknown run_id"):
        registry.record_result("missing", status="COMPLETED")
    with pytest.raises(ValueError, match="status"):
        registry.record_result("missing", status="REGISTERED")

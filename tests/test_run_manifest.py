from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.dataset_provenance import build_dataset_provenance
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


def _fixture_provenance(version="v1"):
    bars = make_bars()
    return build_dataset_provenance(
        source_name="fixture-provider", source_version=version, snapshot_sha256="a" * 64,
        retrieved_at=datetime(2025, 1, 2, tzinfo=timezone.utc), timestamp_timezone="UTC",
        duplicate_policy="reject", adjustment_policy="none", license_or_terms_reference="fixture-only",
        instruments=["EUR/USD"], timeframe="M5", row_count=len(bars),
    )


def test_provenance_is_persisted_and_bound_to_manifest_identity():
    bars = make_bars()
    provenance = _fixture_provenance()
    kwargs = dict(
        code_commit="abc123", model_id="qwen-local", model_revision="revision-1",
        dataset_id="eurusd-m5-fixture-v1",
        evaluation_config={"horizon_bars": 12, "cost_model": "explicit-v1"},
        dataset_provenance=provenance, actual_snapshot_sha256="a" * 64,
    )
    first = build_run_manifest(bars, **kwargs)
    second = build_run_manifest(bars, **kwargs)
    assert first.run_id == second.run_id
    assert first.dataset_provenance_sha256
    assert first.dataset_provenance == provenance.to_dict()
    assert first.to_dict()["dataset_provenance_sha256"] == first.dataset_provenance_sha256


def test_provenance_bound_manifest_rejects_missing_or_mismatched_snapshot():
    bars = make_bars()
    kwargs = dict(
        code_commit="abc123", model_id="qwen-local", model_revision="revision-1",
        dataset_id="fixture", evaluation_config={}, dataset_provenance=_fixture_provenance(),
    )
    with pytest.raises(ValueError, match="actual_snapshot_sha256"):
        build_run_manifest(bars, **kwargs)
    with pytest.raises(ValueError, match="does not match"):
        build_run_manifest(bars, **kwargs, actual_snapshot_sha256="b" * 64)


def test_provenance_changes_change_run_identity():
    bars = make_bars()
    common = dict(
        code_commit="abc123", model_id="qwen-local", model_revision="revision-1",
        dataset_id="eurusd-m5-fixture-v1",
        evaluation_config={"horizon_bars": 12, "cost_model": "explicit-v1"},
        actual_snapshot_sha256="a" * 64,
    )
    first = build_run_manifest(bars, **common, dataset_provenance=_fixture_provenance("v1"))
    changed = build_run_manifest(bars, **common, dataset_provenance=_fixture_provenance("v2"))
    assert first.run_id != changed.run_id

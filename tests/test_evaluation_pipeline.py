from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from app.cost_aware_evaluation import RoundTripCosts
from app.dataset_provenance import build_dataset_provenance
from app.evaluation import EvaluationConfig, ForexBar
from app.evaluation_pipeline import compare_scored_outcomes, prepare_research_run
from app.outcome_labeling import label_outcome
from app.run_manifest import ExperimentRegistry


def make_bars(symbol="EUR/USD", count=120):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(count):
        close = 1.1 + i * 0.0001
        bars.append(ForexBar(
            symbol=symbol, timeframe="M5", timestamp=start + timedelta(minutes=5 * i),
            open=close - 0.00002, high=close + 0.0002,
            low=close - 0.0002, close=close,
            bid=close - 0.00001, ask=close + 0.00001,
        ))
    return bars


def config():
    return EvaluationConfig(
        symbols=("EUR/USD",), timeframe="M5", horizon_bars=2,
        data_source="synthetic-fixture-only", dataset_version="fixture-v1",
        entry_price_rule="explicit test-only entry rule",
        exit_price_rule="explicit test-only exit rule",
        spread_cost_price_units=0.00002, slippage_price_units=0.00001,
        commission_price_units=0.0, target_rule="target distance 0.00015",
        stop_rule="stop distance 0.00015", ambiguous_bar_rule="EXCLUDE",
        primary_metric="paired net price-unit difference",
    )


def _fixture_provenance(bars, version="v1"):
    return build_dataset_provenance(
        source_name="fixture-provider", source_version=version,
        snapshot_sha256="a" * 64, retrieved_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        timestamp_timezone="UTC", duplicate_policy="reject", adjustment_policy="none",
        license_or_terms_reference="fixture-only", instruments=["EUR/USD"],
        timeframe="M5", row_count=len(bars),
    )


def _prepare(registry, bars, **kwargs):
    return prepare_research_run(
        bars, config=config(), code_commit="test-commit", model_id="candidate",
        model_revision="fixture-1", initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, registry=registry, **kwargs,
    )


def test_prepare_registers_reproducible_run_and_walk_forward_plan(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    bars = make_bars()
    first = _prepare(registry, bars)
    second = _prepare(registry, bars)
    assert first.manifest.run_id == second.manifest.run_id
    assert first.registry_inserted is True
    assert second.registry_inserted is False
    assert first.quality_report["invalid_rows"] == 0
    assert first.fold_plans[0].folds


def test_prepare_fails_closed_for_bad_data_or_config_mismatch(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    bars = make_bars()
    bars[4] = bars[3]
    with pytest.raises(ValueError, match="quality checks"):
        _prepare(registry, bars)
    with pytest.raises(ValueError, match="symbols"):
        _prepare(registry, make_bars("GBP/USD"))


def test_prepare_binds_dataset_provenance_into_registered_manifest(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    bars = make_bars()
    provenance = _fixture_provenance(bars)
    result = _prepare(
        registry, bars, dataset_provenance=provenance,
        actual_snapshot_sha256="a" * 64,
    )
    assert result.manifest.dataset_provenance == provenance.to_dict()
    assert result.manifest.dataset_provenance_sha256
    assert registry.get(result.manifest.run_id) is not None


def test_prepare_rejects_partial_or_inconsistent_provenance_before_registration(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    registry = ExperimentRegistry(db_path)
    bars = make_bars()
    with pytest.raises(ValueError, match="supplied together"):
        _prepare(registry, bars, dataset_provenance=_fixture_provenance(bars))
    with pytest.raises(ValueError, match="does not match"):
        _prepare(
            registry, bars, dataset_provenance=_fixture_provenance(bars),
            actual_snapshot_sha256="b" * 64,
        )
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0


def test_provenance_changes_create_distinct_registered_runs(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    bars = make_bars()
    first = _prepare(
        registry, bars, dataset_provenance=_fixture_provenance(bars, "v1"),
        actual_snapshot_sha256="a" * 64,
    )
    second = _prepare(
        registry, bars, dataset_provenance=_fixture_provenance(bars, "v2"),
        actual_snapshot_sha256="a" * 64,
    )
    assert first.manifest.run_id != second.manifest.run_id
    assert registry.get(first.manifest.run_id) is not None
    assert registry.get(second.manifest.run_id) is not None


def test_candidate_baseline_cost_scoring_and_block_uncertainty():
    bars = make_bars(count=12)
    candidate = [
        label_outcome(bars, entry_index=i, side="LONG", target_distance=0.00015,
                      stop_distance=0.00015, horizon_bars=2, ambiguous_policy="EXCLUDE")
        for i in (0, 3, 6, 9)
    ]
    baseline = [
        label_outcome(bars, entry_index=i, side="LONG", target_distance=0.00015,
                      stop_distance=0.00015, horizon_bars=2, ambiguous_policy="EXCLUDE")
        for i in (0, 3, 6, 9)
    ]
    result = compare_scored_outcomes(
        candidate, baseline, costs=RoundTripCosts(0.00002, 0.00001, 0.0),
        block_length=2, bootstrap_resamples=100,
    )
    assert result.candidate_summary.scored_outcomes == result.baseline_summary.scored_outcomes
    assert result.uncertainty.paired_difference_mean == pytest.approx(0.0)
    assert result.uncertainty.paired_difference_ci_low <= 0.0 <= result.uncertainty.paired_difference_ci_high


def test_pairing_mismatch_fails_closed():
    bars = make_bars(count=12)
    candidate = [label_outcome(
        bars, entry_index=0, side="LONG", target_distance=0.00015,
        stop_distance=0.00015, horizon_bars=2, ambiguous_policy="EXCLUDE")]
    baseline = [label_outcome(
        bars, entry_index=3, side="LONG", target_distance=0.00015,
        stop_distance=0.00015, horizon_bars=2, ambiguous_policy="EXCLUDE")]
    with pytest.raises(ValueError, match="paired"):
        compare_scored_outcomes(
            candidate, baseline, costs=RoundTripCosts(0, 0, 0),
            block_length=1, bootstrap_resamples=100,
        )

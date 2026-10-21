from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.causal_ta_features import FEATURE_FAMILIES
from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample
from app.research_experiment_matrix import run_experiment_matrix, write_experiment_matrix_report


FEATURES = tuple(name for family in FEATURE_FAMILIES.values() for name in family)


def sample_data(count=150):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    examples = []
    for i in range(count):
        close = 1.1 + i * 0.0001 + ((i % 7) - 3) * 0.00002
        timestamp = start + timedelta(minutes=15 * i)
        bars.append(ForexBar(
            symbol="EUR/USD", timeframe="M15", timestamp=timestamp,
            open=close - 0.00002, high=close + 0.0002, low=close - 0.0002, close=close,
        ))
        features = {name: float((i % (index + 3)) / (index + 3)) for index, name in enumerate(FEATURES)}
        examples.append(TimedBinaryExample(
            timestamp=timestamp, features=features, label=i % 2,
            label_available_at=timestamp + timedelta(minutes=30),
        ))
    return bars, examples


def test_matrix_requires_experiment_budget_before_running():
    bars, examples = sample_data()
    with pytest.raises(ValueError, match="exceeding max_experiments"):
        run_experiment_matrix(
            bars, examples, feature_names=FEATURES, run_id="matrix-test",
            dataset_sha256="a" * 64, model_types=("xgboost", "lightgbm"),
            initial_train_bars=80, validation_bars=30, test_bars=30, step_bars=30,
            horizon_bars=2, include_ablation=True, max_experiments=2,
        )


def test_matrix_rejects_unknown_model_type():
    bars, examples = sample_data()
    with pytest.raises(ValueError, match="unsupported model types"):
        run_experiment_matrix(
            bars, examples, feature_names=FEATURES, run_id="matrix-test",
            dataset_sha256="a" * 64, model_types=("random_forest",),
            initial_train_bars=80, validation_bars=30, test_bars=30, step_bars=30,
            horizon_bars=2, include_ablation=False,
        )


def test_report_writer_hashes_non_actionable_report(tmp_path):
    from app.research_experiment_matrix import ExperimentMatrixReport
    report = ExperimentMatrixReport(
        schema_version=1, run_id="test", dataset_sha256="a" * 64,
        symbol="EUR/USD", timeframe="M15", feature_names=("return_1",),
        model_types=("xgboost",), fold_count=1, experiments=(),
        common_fold_test_windows=(),
    )
    path = tmp_path / "matrix.json"
    write_experiment_matrix_report(report, path)
    import hashlib, json
    payload = json.loads(path.read_text())
    digest = payload.pop("report_payload_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert payload["actionable"] is False
    assert payload["automatic_winner_selection"] is False

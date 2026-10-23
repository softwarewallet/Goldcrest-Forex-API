from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.causal_ta_features import FEATURE_FAMILIES
from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample
from app.research_experiment_matrix import run_experiment_matrix, write_experiment_matrix_report
from app.research_provenance_auditor import audit_experiment_matrix


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



@dataclass(frozen=True)
class _FakeAuditedFold:
    fold_index: int
    test_start_utc: str
    test_end_utc: str
    best_params: dict[str, int]
    trials_requested: int
    trials_completed: int


def test_matrix_canonicalizes_utc_offsets_before_provenance_audit(monkeypatch):
    bars, examples = sample_data()

    def fake_run_optuna_walk_forward(*args, **kwargs):
        plan = kwargs["plan"]
        folds = tuple(
            _FakeAuditedFold(
                fold_index=fold.fold_index,
                test_start_utc=fold.test[0].timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                test_end_utc=fold.test[-1].timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                best_params={"max_depth": 2},
                trials_requested=kwargs["n_trials_per_fold"],
                trials_completed=kwargs["n_trials_per_fold"],
            )
            for fold in plan.folds
        )
        return SimpleNamespace(
            feature_schema_sha256="b" * 64,
            fold_count=len(folds),
            folds=folds,
        )

    monkeypatch.setattr("app.research_experiment_matrix.run_optuna_walk_forward", fake_run_optuna_walk_forward)
    report = run_experiment_matrix(
        bars, examples, feature_names=FEATURES, run_id="timestamp-audit-test",
        dataset_sha256="a" * 64, model_types=("xgboost",),
        initial_train_bars=80, validation_bars=30, test_bars=30, step_bars=30,
        horizon_bars=2, n_trials_per_fold=2, bootstrap_resamples=100,
        include_ablation=False, max_experiments=1,
    )

    audit = audit_experiment_matrix(report.to_dict())

    assert audit["status"] == "PASS"
    assert audit["errors"] == []
    assert report.common_fold_test_windows[0]["test_start_utc"].endswith("Z")
    assert report.experiments[0]["fold_results"][0]["test_start_utc"].endswith("Z")

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.experiment_report import build_experiment_report, write_experiment_report


@dataclass(frozen=True)
class Metrics:
    sample_count: int
    brier_score: float


@dataclass(frozen=True)
class Fold:
    fold_index: int
    validation: Metrics
    test: Metrics
    actionable: bool = False


@dataclass(frozen=True)
class Comparison:
    sample_count: int
    candidate_name: str
    baseline_name: str
    comparison: dict
    actionable: bool = False


def fixture_report():
    return build_experiment_report(
        manifest={"run_id": "run-001", "dataset_id": "fixture-v1"},
        fold_results={
            "candidate": (
                Fold(1, Metrics(20, 0.19), Metrics(20, 0.21)),
                Fold(0, Metrics(20, 0.18), Metrics(20, 0.22)),
            ),
            "baseline": (Fold(0, Metrics(20, 0.25), Metrics(20, 0.26)),),
        },
        paired_comparisons=[
            Comparison(20, "candidate", "baseline", {"brier_difference": -0.04})
        ],
        exclusions={"unlabeled": 3, "invalid": 1},
        evaluation_contract={"target": "explicitly-provided-binary-label"},
        warnings=["Synthetic fixture; no market-performance inference."],
    )


def test_report_is_deterministic_sorted_and_does_not_pool_fold_metrics():
    first = fixture_report()
    second = fixture_report()
    assert first == second
    assert first["run_id"] == "run-001"
    assert [row["fold_index"] for row in first["fold_results"]["candidate"]] == [0, 1]
    assert "test" in first["fold_results"]["candidate"][0]
    assert "pooled_test_score" not in first
    assert first["interpretation"]["actionable"] is False
    assert len(first["report_sha256"]) == 64


def test_report_writes_canonical_json_and_checks_fingerprint(tmp_path):
    report = fixture_report()
    destination = write_experiment_report(report, tmp_path / "reports" / "run.json")
    assert destination.is_file()
    import json
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved == report
    assert not destination.with_name(destination.name + ".tmp").exists()
    report["run_id"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        write_experiment_report(report, tmp_path / "tampered.json")


@pytest.mark.parametrize("folds", [
    {},
    {"candidate": ()},
    {"candidate": (Fold(0, Metrics(1, 0.2), Metrics(1, 0.2), actionable=True),)},
    {"candidate": (Fold(0, Metrics(1, 0.2), Metrics(1, 0.2)),
                   Fold(0, Metrics(1, 0.2), Metrics(1, 0.2)))},
])
def test_report_rejects_missing_or_unsafe_fold_results(folds):
    with pytest.raises(ValueError):
        build_experiment_report(manifest={"run_id": "run"}, fold_results=folds)


def test_report_rejects_nonfinite_values_and_invalid_exclusion_counts():
    with pytest.raises(ValueError, match="NaN"):
        build_experiment_report(
            manifest={"run_id": "run"},
            fold_results={"m": (Fold(0, Metrics(1, float("nan")), Metrics(1, 0.2)),)},
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_experiment_report(
            manifest={"run_id": "run"},
            fold_results={"m": (Fold(0, Metrics(1, 0.2), Metrics(1, 0.2)),)},
            exclusions={"bad": -1},
        )


def test_report_requires_run_identity_and_non_actionable_comparisons():
    with pytest.raises(ValueError, match="run_id"):
        build_experiment_report(manifest={}, fold_results={"m": (Fold(0, Metrics(1, 0.2), Metrics(1, 0.2)),)})
    with pytest.raises(ValueError, match="actionable=False"):
        build_experiment_report(
            manifest={"run_id": "run"},
            fold_results={"m": (Fold(0, Metrics(1, 0.2), Metrics(1, 0.2)),)},
            paired_comparisons=[Comparison(1, "m", "b", {}, actionable=True)],
        )

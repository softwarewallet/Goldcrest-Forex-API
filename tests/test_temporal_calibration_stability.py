import pytest

from app.temporal_calibration_stability import build_temporal_calibration_stability_audit


def fixture():
    folds = {"model": []}
    boundaries = []
    for i, (ece, low, high) in enumerate([(0.1, 0.08, 0.12), (0.16, 0.13, 0.18), (0.3, 0.25, 0.34)]):
        folds["model"].append({
            "fold_index": i,
            "validation": {"expected_calibration_error": ece, "sample_count": 40,
                           "calibration_uncertainty": {"interval_low": low, "interval_high": high}},
            "test": {"expected_calibration_error": ece, "sample_count": 40,
                     "calibration_uncertainty": {"interval_low": low, "interval_high": high}},
        })
        boundaries.append({"fold_index": i, "test_start_utc": f"2026-0{i+1}-01T00:00:00Z",
                           "test_end_utc": f"2026-0{i+1}-20T00:00:00Z"})
    return folds, boundaries


def test_temporal_calibration_audit_sorts_by_explicit_boundaries_and_flags_endpoint_separation():
    folds, boundaries = fixture()
    folds["model"] = list(reversed(folds["model"]))
    audit = build_temporal_calibration_stability_audit(folds, boundaries)
    test_row = next(row for row in audit["comparisons"] if row["partition"] == "test")
    assert test_row["fold_indices"] == [0, 1, 2]
    assert test_row["first_to_last_ece_delta"] == pytest.approx(0.2)
    assert test_row["status"] == "ENDPOINT_ECE_INCREASE_INTERVALS_DISJOINT"
    assert test_row["actionable"] is False


def test_temporal_audit_withholds_interpretation_without_three_periods():
    folds, boundaries = fixture()
    audit = build_temporal_calibration_stability_audit(
        {"model": folds["model"][:2]}, boundaries[:2]
    )
    assert all(row["status"] == "INSUFFICIENT_TEMPORAL_COVERAGE" for row in audit["comparisons"])


def test_temporal_audit_does_not_infer_order_without_declared_boundaries():
    folds, _ = fixture()
    audit = build_temporal_calibration_stability_audit(folds, [])
    assert all(row["period_count"] == 0 for row in audit["comparisons"])
    assert all(row["status"] == "INSUFFICIENT_TEMPORAL_COVERAGE" for row in audit["comparisons"])


def test_temporal_audit_rejects_duplicate_or_invalid_boundaries():
    folds, boundaries = fixture()
    with pytest.raises(ValueError, match="duplicate fold index"):
        build_temporal_calibration_stability_audit(folds, boundaries + [boundaries[0]])
    broken = [dict(boundaries[0], test_start_utc="2026-01-30T00:00:00Z")]
    with pytest.raises(ValueError, match="start before end"):
        build_temporal_calibration_stability_audit(folds, broken)

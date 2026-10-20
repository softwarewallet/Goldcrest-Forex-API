import pytest
from app.regime_uncertainty_audit import build_regime_uncertainty_audit

def fold(index, brier=0.2, loss=0.5):
    return {"fold_index": index, "validation": {"sample_count": 10},
        "test": {"sample_count": 10, "brier_score": brier, "log_loss": loss,
                 "accuracy_at_fixed_half": 0.6, "expected_calibration_error": 0.1},
        "actionable": False}

def boundaries(count):
    return [{"fold_index": i, "test_start_utc": f"2025-01-{i+1:02d}T00:00:00Z",
             "test_end_utc": f"2025-01-{i+1:02d}T23:59:59Z"} for i in range(count)]

def annotations(labels):
    return {"source": "reviewed-regime-source-v1", "method": "predeclared-protocol-v1",
        "source_snapshot_sha256": "a" * 64,
        "folds": {str(i): {"label": label, "available_at_utc": f"2024-12-{i+1:02d}T00:00:00Z",
            "valid_from_utc": f"2024-12-{i+1:02d}T00:00:00Z",
            "valid_through_utc": f"2025-01-{i+1:02d}T23:59:59Z"} for i, label in enumerate(labels)}}

def test_fold_dispersion_is_descriptive():
    data = {"candidate": [fold(0, .1, .3), fold(1, .2, .5), fold(2, .3, .7)],
            "training_prevalence": [fold(0), fold(1), fold(2)]}
    audit = build_regime_uncertainty_audit(data, [{"candidate_name": "candidate", "baseline_name": "training_prevalence"}])
    assert audit["uncertainty_envelopes"][0]["metrics"]["brier_score"]["median"] == pytest.approx(.2)
    assert audit["confidence_intervals_produced"] is False and audit["actionable"] is False

def test_point_in_time_regimes_are_validated_against_test_windows():
    result = build_regime_uncertainty_audit({"candidate": [fold(0), fold(1), fold(2), fold(3)]},
        [], annotations(["LOW", "LOW", "HIGH", "HIGH"]), boundaries(4))
    regimes = result["regime_annotations"]
    assert regimes["status"] == "DECLARED_AND_TEMPORALLY_VALIDATED"
    assert regimes["temporal_validity_checked"] is True
    assert [r["regime"] for r in regimes["stratified_results"]] == ["HIGH", "LOW"]
    assert all(r["fold_count"] == 2 for r in regimes["stratified_results"])

def test_legacy_labels_are_not_stratified():
    old = {"source": "old-source", "method": "old-method", "folds": {"0": "LOW", "1": "LOW"}}
    result = build_regime_uncertainty_audit({"candidate": [fold(0), fold(1)]}, [], old, boundaries(2))
    assert result["regime_annotations"]["status"] == "LEGACY_LABELS_WITHOUT_TEMPORAL_PROOF"
    assert result["regime_annotations"]["stratified_results"] == []

def test_late_annotation_is_rejected():
    item = annotations(["LOW"])
    item["folds"]["0"]["available_at_utc"] = "2025-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="not available by test start"):
        build_regime_uncertainty_audit({"candidate": [fold(0)]}, [], item, boundaries(1))

def test_validity_must_cover_entire_test_window():
    item = annotations(["LOW"])
    item["folds"]["0"]["valid_through_utc"] = "2025-01-01T12:00:00Z"
    with pytest.raises(ValueError, match="does not cover the entire test window"):
        build_regime_uncertainty_audit({"candidate": [fold(0)]}, [], item, boundaries(1))

def test_source_snapshot_hash_is_validated():
    item = annotations(["LOW"])
    item["source_snapshot_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="source_snapshot_sha256"):
        build_regime_uncertainty_audit({"candidate": [fold(0)]}, [], item, boundaries(1))

def test_temporal_boundaries_require_explicit_utc():
    bad = [{"fold_index": 0, "test_start_utc": "2025-01-01T00:00:00",
            "test_end_utc": "2025-01-01T23:00:00Z"}]
    with pytest.raises(ValueError, match="explicitly use UTC"):
        build_regime_uncertainty_audit({"candidate": [fold(0)]}, [], None, bad)

def test_missing_temporal_boundaries_withholds_regimes():
    result = build_regime_uncertainty_audit({"candidate": [fold(0)]}, [], annotations(["LOW"]), None)
    assert result["regime_annotations"]["status"] == "WITHHELD_MISSING_FOLD_TEMPORAL_BOUNDARIES"
    assert result["regime_annotations"]["stratified_results"] == []

def test_only_common_complete_folds():
    data = {"a": [fold(0), fold(1), fold(2)], "b": [fold(0), fold(1, float("nan")), fold(2)]}
    assert build_regime_uncertainty_audit(data, [])["common_complete_test_fold_indices"] == [0, 2]

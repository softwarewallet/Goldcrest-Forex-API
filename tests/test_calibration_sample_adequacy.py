from app.calibration_sample_adequacy import build_calibration_sample_adequacy_audit


def test_calibration_sample_audit_flags_small_samples_and_minor_class():
    audit = build_calibration_sample_adequacy_audit({"candidate": [
        {"fold_index": 0, "validation": {"sample_count": 20, "positive_count": 10, "negative_count": 10},
         "test": {"sample_count": 50, "positive_count": 4, "negative_count": 46}}]})
    validation, test = audit["comparisons"]
    assert validation["status"] == "LOW_SAMPLE_COUNT"
    assert test["status"] == "LOW_MINOR_CLASS_COUNT"
    assert audit["actionable"] is False


def test_calibration_sample_audit_reports_missing_class_counts_without_guessing():
    audit = build_calibration_sample_adequacy_audit({"candidate": [
        {"fold_index": 0, "validation": {"sample_count": 40}, "test": {"sample_count": 40}}]})
    assert audit["comparisons"][0]["positive_count"] is None
    assert audit["comparisons"][0]["negative_count"] is None
    assert audit["comparisons"][0]["status"] == "CLASS_COUNTS_UNAVAILABLE_OR_INCONSISTENT"


def test_calibration_sample_audit_flags_sparse_populated_reliability_bins():
    bins = [{"count": count} for count in (12, 8, 3, 0)]
    audit = build_calibration_sample_adequacy_audit({"candidate": [{"fold_index": 1,
        "validation": {"sample_count": 40, "positive_count": 20, "negative_count": 20, "reliability_bins": bins},
        "test": {"sample_count": 40, "positive_count": 20, "negative_count": 20, "reliability_bins": bins}}]})
    assert audit["comparisons"][0]["populated_reliability_bins"] == 3
    assert audit["comparisons"][0]["sparse_populated_reliability_bins"] == 2
    assert audit["comparisons"][0]["status"] == "SPARSE_RELIABILITY_BINS"

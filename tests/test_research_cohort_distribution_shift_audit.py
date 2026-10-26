from __future__ import annotations

import hashlib

import pytest

from app.research_cohort_distribution_shift_audit import audit_cohort_distribution_shift, _load_json_object


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest():
    records = []
    for seed in (11, 22, 33):
        for fold, start, end in (
            ("fold_1", "2025-01-01", "2025-02-01"),
            ("fold_2", "2025-02-02", "2025-03-01"),
        ):
            records.append({
                "run_id": f"{seed}_{fold}",
                "seed": seed,
                "fold_id": fold,
                "symbol": "EURUSD",
                "timeframe": "M15",
                "feature_schema_sha256": sha(b"features-v1"),
                "strategy_config_sha256": sha(b"config-v1"),
                "dataset_sha256": sha(f"data-{fold}".encode()),
                "window_start": start,
                "window_end": end,
            })
    return {
        "expected_folds": ["fold_1", "fold_2"],
        "runs": records,
        "distribution_summaries": [{
            "feature": "rsi_14",
            "reference": {
                "mean": 50.0, "std": 10.0,
                "histogram": {"edges": [0, 25, 50, 75, 100], "counts": [10, 30, 40, 20]},
            },
            "evaluation": {
                "mean": 52.0, "std": 11.0,
                "histogram": {"edges": [0, 25, 50, 75, 100], "counts": [8, 28, 42, 22]},
            },
        }],
    }


def audit(data, **kwargs):
    return audit_cohort_distribution_shift(
        data, manifest_sha256=sha(b"manifest"), **kwargs
    )


def codes(report):
    return {item["code"] for item in report["errors"]}


def test_comparable_cohort_and_low_shift_pass():
    report = audit(manifest())
    assert report["status"] == "PASS"
    assert report["runs_supplied"] == report["runs_validated"] == 6
    assert report["distribution_shift"]["features_assessed"] == 1
    assert "population_stability_index" in report["distribution_shift"]["feature_results"][0]
    assert report["execution_connected"] is False
    assert report["automatic_model_selection"] is False


def test_feature_schema_and_strategy_config_mismatch_fail():
    data = manifest()
    data["runs"][0]["feature_schema_sha256"] = sha(b"different-features")
    data["runs"][1]["strategy_config_sha256"] = sha(b"different-config")
    report = audit(data)
    assert "cohort_field_mismatch" in codes(report)


def test_dataset_hash_and_window_must_match_within_fold():
    data = manifest()
    data["runs"][1]["dataset_sha256"] = sha(b"other-data")
    data["runs"][3]["window_start"] = "2025-02-05"
    report = audit(data)
    assert "dataset_fingerprint_mismatch_within_fold" in codes(report)
    assert "evaluation_window_mismatch_within_fold" in codes(report)


def test_missing_expected_fold_fails():
    data = manifest()
    data["runs"] = [run for run in data["runs"] if run["fold_id"] != "fold_2"]
    report = audit(data)
    assert "expected_folds_missing" in codes(report)


def test_duplicate_run_and_seed_fold_fail():
    data = manifest()
    data["runs"].append(dict(data["runs"][0]))
    report = audit(data)
    assert "duplicate_run_id" in codes(report)
    assert "duplicate_seed_fold" in codes(report)


def test_high_psi_and_mean_shift_raise_warnings():
    data = manifest()
    data["distribution_summaries"][0]["evaluation"] = {
        "mean": 80.0, "std": 15.0,
        "histogram": {"edges": [0, 25, 50, 75, 100], "counts": [70, 10, 5, 5]},
    }
    report = audit(data)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "population_stability_index_high:rsi_14" in report["warnings"]
    assert "standardized_mean_shift_high:rsi_14" in report["warnings"]


def test_invalid_distribution_statistics_and_histogram_fail():
    data = manifest()
    data["distribution_summaries"][0]["reference"]["std"] = -1
    second = {
        "feature": "atr_14",
        "reference": {
            "mean": 1.0, "std": 0.5,
            "histogram": {"edges": [0, 1, 2], "counts": [10, 20]},
        },
        "evaluation": {
            "mean": 1.2, "std": 0.6,
            "histogram": {"edges": [0, 2, 1], "counts": [10, 20]},
        },
    }
    data["distribution_summaries"].append(second)
    report = audit(data)
    assert "distribution_std_negative" in codes(report)
    assert "histogram_invalid_or_incompatible" in codes(report)


def test_absent_distribution_summaries_warn_not_silent_pass():
    data = manifest()
    data.pop("distribution_summaries")
    report = audit(data)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "distribution_summaries_not_supplied; shift was not assessed" in report["warnings"]


def test_bad_thresholds_and_manifest_hash_rejected():
    with pytest.raises(ValueError, match="psi_warning_threshold"):
        audit(manifest(), psi_warning_threshold=-0.1)
    with pytest.raises(ValueError, match="manifest_sha256_invalid"):
        audit_cohort_distribution_shift(manifest(), manifest_sha256="bad")


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_strict_json_loader_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="cohort-manifest.json")

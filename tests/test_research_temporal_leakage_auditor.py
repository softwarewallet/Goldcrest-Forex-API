import hashlib
import json

from app.research_temporal_leakage_auditor import audit_temporal_integrity


def _seal(payload):
    # Recompute over the payload excluding any previous digest, matching the auditor contract.
    payload.pop("report_payload_sha256", None)
    payload["report_payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def _report():
    return _seal({
        "run_id": "temporal-test",
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "common_fold_test_windows": [{
            "fold_index": 0,
            "test_start_utc": "2026-01-01T03:00:00Z",
            "test_end_utc": "2026-01-01T04:00:00Z",
        }],
        "experiments": [{
            "actionable": False,
            "test_metrics_used_for_tuning": False,
            "fold_results": [{
                "fold_index": 0,
                "train_start_utc": "2026-01-01T00:00:00Z",
                "train_end_utc": "2026-01-01T00:59:00Z",
                "validation_start_utc": "2026-01-01T01:15:00Z",
                "validation_end_utc": "2026-01-01T01:59:00Z",
                "test_start_utc": "2026-01-01T03:00:00Z",
                "test_end_utc": "2026-01-01T04:00:00Z",
                "train_rows_excluded_unmatured": 2,
                "refit_rows_excluded_unmatured": 1,
            }],
        }],
    })


def test_temporal_audit_passes_chronological_report():
    result = audit_temporal_integrity(_report())
    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_temporal_audit_rejects_validation_test_overlap():
    report = _report()
    report["experiments"][0]["fold_results"][0]["validation_end_utc"] = "2026-01-01T03:10:00Z"
    _seal(report)
    result = audit_temporal_integrity(report)
    assert result["status"] == "FAIL"
    assert "experiment_0_fold_0_partition_boundaries_overlap_or_out_of_order" in result["errors"]


def test_temporal_audit_rejects_test_window_mismatch_and_bad_count():
    report = _report()
    fold = report["experiments"][0]["fold_results"][0]
    fold["test_start_utc"] = "2026-01-01T03:01:00Z"
    fold["train_rows_excluded_unmatured"] = -1
    _seal(report)
    result = audit_temporal_integrity(report)
    assert "experiment_0_fold_0_test_window_differs_from_common_plan" in result["errors"]
    assert "experiment_0_fold_0_invalid_train_rows_excluded_unmatured" in result["errors"]


def test_temporal_audit_rejects_timezone_naive_boundaries():
    report = _report()
    report["experiments"][0]["fold_results"][0]["train_start_utc"] = "2026-01-01T00:00:00"
    _seal(report)
    result = audit_temporal_integrity(report)
    assert result["status"] == "FAIL"
    assert "experiment_0_fold_0_invalid_or_timezone_naive_boundary" in result["errors"]


def test_temporal_audit_detects_report_payload_tampering():
    report = _report()
    report["experiments"][0]["fold_results"][0]["train_rows_excluded_unmatured"] = 99
    result = audit_temporal_integrity(report)
    assert "report_payload_sha256_mismatch" in result["errors"]

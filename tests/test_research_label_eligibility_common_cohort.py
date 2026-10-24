from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_label_eligibility_common_cohort import audit_common_heldout_cohort


def _report(path: Path, cap: float, rows: list[dict], *, symbol: str = "EURUSD") -> Path:
    timestamps = sorted(row["timestamp_utc"] for row in rows)
    fingerprint = hashlib.sha256(("\n".join(timestamps) + ("\n" if timestamps else "")).encode()).hexdigest()
    report = {
        "status": "PASS",
        "baseline_run_id": "baseline-365",
        "variant_run_id": f"cap-{cap}",
        "symbol": symbol,
        "timeframe": "M15",
        "label_horizon_bars": 4,
        "raw_ohlc_sha256": "a" * 64,
        "max_label_elapsed_hours": cap,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "experiments": [{
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "fold_results": [{
                "fold_index": 0,
                "test_start_utc": "2026-01-01T00:00:00Z",
                "test_end_utc": "2026-01-01T02:00:00Z",
                "eligible_timestamp_fingerprint_sha256": fingerprint,
                "eligible_timestamp_predictions": rows,
            }],
        }],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _rows(prob1: float = 0.8) -> list[dict]:
    return [
        {"timestamp_utc": "2026-01-01T00:00:00Z", "label": 0,
         "baseline_probability_class_1": 0.2, "variant_probability_class_1": 0.25},
        {"timestamp_utc": "2026-01-01T01:00:00Z", "label": 1,
         "baseline_probability_class_1": 0.8, "variant_probability_class_1": prob1},
    ]


def test_common_cohort_recomputes_metrics_and_keeps_safety_flags(tmp_path):
    rows_a = _rows(0.75)
    rows_b = _rows(0.85)
    rows_b.append({
        "timestamp_utc": "2026-01-01T02:00:00Z", "label": 1,
        "baseline_probability_class_1": 0.9, "variant_probability_class_1": 0.95,
    })
    a = _report(tmp_path / "cap1.json", 1.0, rows_a)
    b = _report(tmp_path / "cap8.json", 8.0, rows_b)

    result = audit_common_heldout_cohort([a, b])

    assert result["status"] == "PASS"
    assert result["caps_hours"] == [1.0, 8.0]
    assert result["automatic_model_selection"] is False
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    fold = result["experiments"][0]["fold_results"][0]
    assert fold["common_timestamp_count"] == 2
    assert fold["per_cap_eligible_timestamp_count"] == {"1.0": 2, "8.0": 3}
    assert fold["per_cap_excluded_to_common_count"] == {"1.0": 0, "8.0": 1}
    assert fold["baseline_metrics_on_common_cohort"]["sample_count"] == 2
    assert fold["variant_metrics_on_common_cohort_by_cap"]["1.0"]["sample_count"] == 2
    assert any("does not imply identical training data" in warning for warning in result["warnings"])


def test_common_cohort_fails_on_incompatible_provenance(tmp_path):
    a = _report(tmp_path / "cap1.json", 1.0, _rows())
    b = _report(tmp_path / "cap8.json", 8.0, _rows(), symbol="GBPUSD")

    result = audit_common_heldout_cohort([a, b])

    assert result["status"] == "FAIL"
    assert any("symbol" in error for error in result["errors"])
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_common_cohort_fails_on_bad_timestamp_fingerprint(tmp_path):
    a = _report(tmp_path / "cap1.json", 1.0, _rows())
    b = _report(tmp_path / "cap8.json", 8.0, _rows())
    payload = json.loads(b.read_text())
    payload["experiments"][0]["fold_results"][0]["eligible_timestamp_fingerprint_sha256"] = "0" * 64
    b.write_text(json.dumps(payload))

    result = audit_common_heldout_cohort([a, b])

    assert result["status"] == "FAIL"
    assert any("fingerprint mismatch" in error for error in result["errors"])
    assert result["automatic_model_selection"] is False


def test_common_cohort_fails_when_labels_disagree_on_shared_timestamp(tmp_path):
    a = _report(tmp_path / "cap1.json", 1.0, _rows())
    altered = _rows()
    altered[1]["label"] = 0
    b = _report(tmp_path / "cap8.json", 8.0, altered)

    result = audit_common_heldout_cohort([a, b])

    assert result["status"] == "FAIL"
    assert any("label mismatch" in error for error in result["errors"])
    assert result["actionable"] is False


def test_common_cohort_fails_when_no_timestamp_records_are_present(tmp_path):
    a = _report(tmp_path / "cap1.json", 1.0, [])
    b = _report(tmp_path / "cap8.json", 8.0, _rows())

    result = audit_common_heldout_cohort([a, b])

    assert result["status"] == "FAIL"
    assert result["actionable"] is False
    assert result["execution_connected"] is False

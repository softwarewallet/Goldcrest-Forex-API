from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_label_eligibility_cross_report_integrity import (
    audit_cross_report_integrity,
    main,
)


def _report(path: Path, cap: float, *, symbol: str = "EURUSD") -> Path:
    rows = [
        {"timestamp_utc": "2026-01-01T00:00:00Z", "label": 0,
         "baseline_probability_class_1": 0.2, "variant_probability_class_1": 0.25},
        {"timestamp_utc": "2026-01-01T01:00:00Z", "label": 1,
         "baseline_probability_class_1": 0.8, "variant_probability_class_1": 0.75},
    ]
    fingerprint = hashlib.sha256(
        ("\\n".join(row["timestamp_utc"] for row in rows) + "\\n").encode()
    ).hexdigest()
    report = {
        "status": "PASS",
        "baseline_run_id": "baseline-1",
        "baseline_dataset_sha256": "b" * 64,
        "variant_run_id": f"cap-{cap}",
        "symbol": symbol,
        "timeframe": "M15",
        "label_horizon_bars": 4,
        "raw_ohlc_sha256": "a" * 64,
        "max_label_elapsed_hours": cap,
        "inferential": False,
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


def test_cross_report_integrity_records_exact_input_hashes_and_identity(tmp_path):
    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0)

    result = audit_cross_report_integrity([str(first), str(second)])

    assert result["status"] == "PASS"
    receipts = result["input_report_receipts"]
    assert [item["sha256"] for item, path in zip(receipts, [first, second])] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in [first, second]
    ]
    assert [item["max_label_elapsed_hours"] for item in receipts] == [1.0, 8.0]
    assert result["automatic_model_selection"] is False
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_cross_report_integrity_fails_on_provenance_mismatch(tmp_path):
    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0, symbol="GBPUSD")

    result = audit_cross_report_integrity([str(first), str(second)])

    assert result["status"] == "FAIL"
    assert any("common-cohort audit failed" in error for error in result["errors"])
    assert result["actionable"] is False


def test_cross_report_integrity_cli_refuses_input_alias_and_existing_output(tmp_path, capsys):
    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0)

    assert main(["--comparison-report", str(first), "--comparison-report", str(second),
                 "--output", str(first)]) == 2
    assert "output_must_not_alias_input" in capsys.readouterr().err

    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    assert main(["--comparison-report", str(first), "--comparison-report", str(second),
                 "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "preserve"

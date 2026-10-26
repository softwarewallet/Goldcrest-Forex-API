from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_label_eligibility_cross_report_integrity import (
    audit_cross_report_integrity,
    main,
    _write_report_no_clobber,
)


def _report(path: Path, cap: float, *, symbol: str = "EURUSD") -> Path:
    rows = [
        {"timestamp_utc": "2026-01-01T00:00:00Z", "label": 0,
         "baseline_probability_class_1": 0.2, "variant_probability_class_1": 0.25},
        {"timestamp_utc": "2026-01-01T01:00:00Z", "label": 1,
         "baseline_probability_class_1": 0.8, "variant_probability_class_1": 0.75},
    ]
    fingerprint = hashlib.sha256(
        ("\n".join(row["timestamp_utc"] for row in rows) + "\n").encode()
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



def test_cross_report_integrity_detects_input_mutation_during_audit(tmp_path, monkeypatch):
    import app.research_label_eligibility_cross_report_integrity as module

    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0)
    real_audit = module.audit_common_heldout_cohort

    def mutate_after_audit(paths, *, allow_baseline_id_variation=False):
        result = real_audit(paths, allow_baseline_id_variation=allow_baseline_id_variation)
        payload = json.loads(second.read_text(encoding="utf-8"))
        payload["variant_run_id"] = "tampered-after-audit"
        second.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr(module, "audit_common_heldout_cohort", mutate_after_audit)
    result = audit_cross_report_integrity([str(first), str(second)])

    assert result["status"] == "FAIL"
    assert any("changed during audit" in error for error in result["errors"])
    assert result["actionable"] is False


def test_atomic_writer_refuses_competing_destination_without_clobber(tmp_path, monkeypatch):
    import app.research_label_eligibility_cross_report_integrity as module
    import pytest

    output = tmp_path / "audit.json"
    real_link = module.os.link

    def create_destination_then_link(source, destination):
        Path(destination).write_text("created by competing writer", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", create_destination_then_link)
    with pytest.raises(FileExistsError):
        _write_report_no_clobber(output, {"status": "PASS"}, [])
    assert output.read_text(encoding="utf-8") == "created by competing writer"
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []


def test_cross_report_integrity_rejects_malformed_provenance_digest(tmp_path):
    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["raw_ohlc_sha256"] = "not-a-sha256"
    second.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_cross_report_integrity([str(first), str(second)])

    assert result["status"] == "FAIL"
    assert any("invalid_or_missing_raw_ohlc_sha256" in error for error in result["errors"])
    assert result["actionable"] is False


def test_cross_report_integrity_rejects_duplicate_input_paths_and_variant_ids(tmp_path):
    first = _report(tmp_path / "cap1.json", 1.0)
    second = _report(tmp_path / "cap8.json", 8.0)

    duplicate_path = audit_cross_report_integrity([str(first), str(first)])
    assert duplicate_path["status"] == "FAIL"
    assert any("duplicate_input_report_path" in error for error in duplicate_path["errors"])

    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["variant_run_id"] = "cap-1.0"
    second.write_text(json.dumps(payload), encoding="utf-8")
    duplicate_variant = audit_cross_report_integrity([str(first), str(second)])
    assert duplicate_variant["status"] == "FAIL"
    assert any("duplicate_variant_run_id" in error for error in duplicate_variant["errors"])

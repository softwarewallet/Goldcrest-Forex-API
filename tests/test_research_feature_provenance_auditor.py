import hashlib
import json
from pathlib import Path

from app.research_feature_provenance_auditor import audit_feature_provenance


def _manifest(dataset, features):
    payload = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
        "actionable": False,
        "automatic_sorting": False,
        "automatic_imputation": False,
        "feature_names": features,
        "feature_schema_sha256": hashlib.sha256(json.dumps({"feature_names": features}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        "feature_implementation_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "app" / "causal_ta_features.py").read_bytes()).hexdigest(),
        "feature_implementation_file": "app/causal_ta_features.py",
        "feature_timestamp_rule": "Each feature at t uses only OHLC observations timestamped <= t.",
        "output_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def test_feature_provenance_passes_with_structural_warning(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.01\n"
        "EURUSD,M15,2026-01-01T00:15:00Z,1.05,1.2,1,1.1,0,2026-01-01T01:15:00Z,-0.02\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(dataset, ["return_1"])), encoding="utf-8")
    result = audit_feature_provenance(dataset, manifest)
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert result["row_count"] == 2
    assert result["feature_count"] == 1
    assert result["actionable"] is False


def test_feature_provenance_rejects_label_timestamp_overlap(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T00:00:00Z,0.01\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(dataset, ["return_1"])), encoding="utf-8")
    result = audit_feature_provenance(dataset, manifest)
    assert result["status"] == "FAIL"
    assert "row_2_label_not_matured_after_timestamp" in result["errors"]


def test_feature_provenance_detects_dataset_tampering(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.01\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(dataset, ["return_1"])), encoding="utf-8")
    dataset.write_text(dataset.read_text(encoding="utf-8") + "EURUSD,M15,2026-01-01T00:15:00Z,1,1.1,0.9,1.05,0,2026-01-01T01:15:00Z,0.02\n", encoding="utf-8")
    result = audit_feature_provenance(dataset, manifest)
    assert result["status"] == "FAIL"
    assert "dataset_sha256_mismatch" in result["errors"]


def test_feature_provenance_detects_schema_fingerprint_mismatch(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.01\n",
        encoding="utf-8",
    )
    payload = _manifest(dataset, ["return_1"])
    payload["feature_schema_sha256"] = "0" * 64
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = audit_feature_provenance(dataset, manifest)
    assert result["status"] == "FAIL"
    assert "feature_schema_sha256_mismatch" in result["errors"]


def test_feature_provenance_detects_source_hash_mismatch(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.01\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(dataset, ["return_1"])), encoding="utf-8")
    altered_source = tmp_path / "causal_ta_features.py"
    altered_source.write_text("# altered feature implementation\n", encoding="utf-8")
    result = audit_feature_provenance(dataset, manifest, feature_source_path=altered_source)
    assert result["status"] == "FAIL"
    assert "feature_implementation_sha256_mismatch" in result["errors"]

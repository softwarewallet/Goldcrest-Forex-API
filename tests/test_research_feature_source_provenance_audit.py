from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from app.research_feature_source_provenance_audit import audit_feature_source_provenance


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _artifacts(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
                "label", "label_available_at", "ret_1",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({
            "symbol": "EURUSD",
            "timeframe": "M15",
            "timestamp": "2026-01-01T00:00:00Z",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "label": 1,
            "label_available_at": "2026-01-01T01:00:00Z",
            "ret_1": 0.1,
        })
    source = Path(__file__).resolve().parents[1] / "app" / "causal_ta_features.py"
    names = ["ret_1"]
    manifest = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
        "actionable": False,
        "output_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "output_rows": 1,
        "feature_names": names,
        "feature_schema_sha256": hashlib.sha256(_canonical({"feature_names": names})).hexdigest(),
        "feature_implementation_file": "app/causal_ta_features.py",
        "feature_implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path = tmp_path / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return dataset, manifest_path, source, manifest


def test_feature_source_provenance_passes_for_matching_dataset_manifest_and_source(tmp_path):
    dataset, manifest_path, source, _ = _artifacts(tmp_path)

    report = audit_feature_source_provenance(dataset, manifest_path, feature_source_path=source)

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["feature_count"] == 1
    assert report["dataset_rows"] == 1
    assert report["actionable"] is False
    assert report["execution_connected"] is False
    assert report["model_training_performed"] is False


def test_feature_source_provenance_rejects_dataset_tampering(tmp_path):
    dataset, manifest_path, source, _ = _artifacts(tmp_path)
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_feature_source_provenance(dataset, manifest_path, feature_source_path=source)

    assert report["status"] == "FAIL"
    assert "output_dataset_sha256_mismatch" in report["errors"]


def test_feature_source_provenance_rejects_source_hash_mismatch(tmp_path):
    dataset, manifest_path, source, manifest = _artifacts(tmp_path)
    manifest["feature_implementation_sha256"] = "0" * 64
    manifest["manifest_sha256"] = hashlib.sha256(_canonical({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    })).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    report = audit_feature_source_provenance(dataset, manifest_path, feature_source_path=source)

    assert report["status"] == "FAIL"
    assert "feature_implementation_sha256_mismatch" in report["errors"]


def test_feature_source_provenance_rejects_schema_and_manifest_tampering(tmp_path):
    dataset, manifest_path, source, manifest = _artifacts(tmp_path)
    manifest["feature_names"] = ["different_feature"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    report = audit_feature_source_provenance(dataset, manifest_path, feature_source_path=source)

    assert report["status"] == "FAIL"
    assert "manifest_sha256_mismatch" in report["errors"]
    assert "feature_schema_sha256_mismatch" in report["errors"]
    assert "dataset_feature_columns_mismatch" in report["errors"]

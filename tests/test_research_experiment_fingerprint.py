from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.research_experiment_fingerprint import (
    build_experiment_fingerprint,
    canonical_json,
    verify_experiment_fingerprint,
)


def _manifest_and_paths(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("timestamp,label,ret_1\n2026-01-01T00:00:00Z,1,0.2\n", encoding="utf-8")
    source = tmp_path / "causal_features.py"
    source.write_text("def features(row): return row['close']\n", encoding="utf-8")
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
        "actionable": False,
        "output_dataset_sha256": dataset_hash,
        "feature_implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_schema_sha256": hashlib.sha256(canonical_json({"feature_names": ["ret_1"]})).hexdigest(),
        "feature_names": ["ret_1"],
        "feature_backend": "python",
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest_path = tmp_path / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    implementation = tmp_path / "runner.py"
    implementation.write_text("print('research only')\n", encoding="utf-8")
    return dataset_hash, manifest_path, manifest, {"app/runner.py": implementation}


def _build(tmp_path: Path, **overrides):
    dataset_hash, manifest_path, manifest, implementation_paths = _manifest_and_paths(tmp_path)
    kwargs = {
        "configuration": {"symbol": "EURUSD", "timeframe": "M15", "seed": 7},
        "raw_ohlc_sha256": "a" * 64,
        "dataset_sha256": dataset_hash,
        "dataset_manifest_path": manifest_path,
        "dataset_manifest": manifest,
        "implementation_paths": implementation_paths,
    }
    kwargs.update(overrides)
    return build_experiment_fingerprint(**kwargs)


def test_experiment_fingerprint_is_deterministic_and_verifiable(tmp_path):
    receipt = _build(tmp_path)

    verification = verify_experiment_fingerprint(receipt)

    assert verification["status"] == "PASS"
    assert verification["errors"] == []
    assert receipt["actionable"] is False
    assert receipt["execution_connected"] is False
    assert receipt["automatic_model_selection"] is False


def test_experiment_fingerprint_changes_when_configuration_changes(tmp_path):
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second", configuration={"symbol": "EURUSD", "timeframe": "M15", "seed": 8})

    assert first["experiment_fingerprint_sha256"] != second["experiment_fingerprint_sha256"]


def test_experiment_fingerprint_verifier_rejects_tampered_configuration(tmp_path):
    receipt = _build(tmp_path)
    receipt["configuration"]["seed"] = 99

    verification = verify_experiment_fingerprint(receipt)

    assert verification["status"] == "FAIL"
    assert "experiment_fingerprint_sha256_mismatch" in verification["errors"]


def test_experiment_fingerprint_rejects_invalid_dataset_manifest_self_hash(tmp_path):
    dataset_hash, manifest_path, manifest, implementation_paths = _manifest_and_paths(tmp_path)
    manifest["manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="dataset_manifest_self_hash_mismatch"):
        build_experiment_fingerprint(
            configuration={"seed": 1},
            raw_ohlc_sha256="a" * 64,
            dataset_sha256=dataset_hash,
            dataset_manifest_path=manifest_path,
            dataset_manifest=manifest,
            implementation_paths=implementation_paths,
        )


def test_experiment_fingerprint_rejects_noncanonical_configuration(tmp_path):
    with pytest.raises(ValueError, match="fingerprint_payload_not_canonicalizable"):
        _build(tmp_path, configuration={"seed": float("nan")})

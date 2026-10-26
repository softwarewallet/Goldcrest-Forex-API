from __future__ import annotations

import hashlib

import pytest

from app.research_artifact_freshness_audit import (
    audit_artifact_freshness,
    fingerprint_path,
    _read_object,
)

REV = "a" * 40
OTHER_REV = "b" * 40
CONFIG = hashlib.sha256(b"config").hexdigest()
DATASET = hashlib.sha256(b"dataset").hexdigest()


def artifact(artifact_id, **overrides):
    result = {
        "artifact_id": artifact_id,
        "path": f"{artifact_id}.json",
        "source_revision": REV,
        "config_sha256": CONFIG,
        "dataset_sha256": DATASET,
        "depends_on": [],
    }
    result.update(overrides)
    return result


def manifest(*artifacts):
    return {"audit_name": "research_artifact_freshness_manifest", "artifacts": list(artifacts)}


def audit(payload):
    return audit_artifact_freshness(
        payload,
        current_revision=REV,
        current_config_sha256=CONFIG,
        current_dataset_sha256=DATASET,
    )


def by_id(report):
    return {item["artifact_id"]: item for item in report["artifacts"]}


def test_all_matching_provenance_is_current():
    report = audit(manifest(artifact("plan"), artifact("metrics", depends_on=["plan"])))
    assert report["status"] == "PASS"
    assert report["current_count"] == 2
    assert report["regeneration_order"] == []
    assert report["deployment_authorized"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("source_revision", OTHER_REV, "STALE_CODE_REVISION"),
        ("config_sha256", "c" * 64, "STALE_CONFIG"),
        ("dataset_sha256", "d" * 64, "STALE_DATASET"),
        ("source_revision", None, "MISSING_OR_INVALID_SOURCE_REVISION"),
        ("config_sha256", None, "MISSING_OR_INVALID_CONFIG_FINGERPRINT"),
        ("dataset_sha256", None, "MISSING_OR_INVALID_DATASET_FINGERPRINT"),
        ("path", "", "MISSING_ARTIFACT_PATH"),
    ],
)
def test_stale_or_missing_provenance_requires_regeneration(field, value, reason):
    item = artifact("report", **{field: value})
    report = audit(manifest(item))
    result = by_id(report)["report"]
    assert report["status"] == "STALE"
    assert result["needs_regeneration"] is True
    assert reason in result["reasons"]


def test_staleness_propagates_and_regeneration_order_is_dependency_first():
    report = audit(manifest(
        artifact("plan", source_revision=OTHER_REV),
        artifact("registry", depends_on=["plan"]),
        artifact("metrics", depends_on=["registry"]),
        artifact("unrelated"),
    ))
    results = by_id(report)
    assert results["plan"]["classification"] == "REGENERATION_REQUIRED"
    assert "STALE_DEPENDENCY" in results["registry"]["reasons"]
    assert "STALE_DEPENDENCY" in results["metrics"]["reasons"]
    assert report["regeneration_order"] == ["plan", "registry", "metrics"]
    assert results["unrelated"]["classification"] == "CURRENT"


def test_rejects_duplicate_ids_and_unknown_dependencies():
    with pytest.raises(ValueError, match="duplicate_artifact_id"):
        audit(manifest(artifact("x"), artifact("x")))
    with pytest.raises(ValueError, match="unknown_dependency"):
        audit(manifest(artifact("x", depends_on=["missing"])))


def test_rejects_dependency_cycles():
    with pytest.raises(ValueError, match="dependency_cycle"):
        audit(manifest(artifact("x", depends_on=["y"]), artifact("y", depends_on=["x"])))


def test_rejects_invalid_current_fingerprints():
    with pytest.raises(ValueError, match="current_revision"):
        audit_artifact_freshness(manifest(), current_revision="short", current_config_sha256=CONFIG, current_dataset_sha256=DATASET)
    with pytest.raises(ValueError, match="current_config_sha256"):
        audit_artifact_freshness(manifest(), current_revision=REV, current_config_sha256="invalid", current_dataset_sha256=DATASET)


def test_directory_fingerprint_is_stable_and_sensitive_to_content(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "b.csv").write_text("b", encoding="utf-8")
    (root / "a.csv").write_text("a", encoding="utf-8")
    first = fingerprint_path(root)
    assert first == fingerprint_path(root)
    (root / "a.csv").write_text("changed", encoding="utf-8")
    assert fingerprint_path(root) != first


def test_file_fingerprint_matches_sha256(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"x":1}')
    assert fingerprint_path(path) == hashlib.sha256(b'{"x":1}').hexdigest()


def test_fingerprint_rejects_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint_path(tmp_path / "missing")


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_json_reader_rejects_duplicate_keys_and_nonstandard_constants(tmp_path, raw):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_artifact_provenance import capture_provenance, verify_provenance

REV = "a" * 40
OTHER_REV = "b" * 40


def setup_tree(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "plan.json").write_text('{"plan":1}', encoding="utf-8")
    (root / "metrics.csv").write_text("value\n1\n2\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"window":15}', encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "prices.csv").write_text("close\n1.2\n", encoding="utf-8")
    spec = {"artifacts": [
        {"artifact_id": "plan", "path": "plan.json", "depends_on": []},
        {"artifact_id": "metrics", "path": "metrics.csv", "depends_on": ["plan"]},
    ]}
    return root, config, dataset, spec


def capture(root, config, dataset, spec):
    return capture_provenance(spec, source_revision=REV, config_path=config, dataset_path=dataset, root=root)


def verify(manifest, root, config, dataset, revision=REV):
    return verify_provenance(manifest, root=root, current_source_revision=revision, config_path=config, dataset_path=dataset)


def test_capture_and_verify_current_artifacts(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    assert manifest["audit_name"] == "research_artifact_provenance_manifest"
    assert manifest["artifact_count"] == 2
    report = verify(manifest, root, config, dataset)
    assert report["status"] == "PASS"
    assert report["current_count"] == 2
    assert report["deployment_authorized"] is False


def test_changed_output_bytes_are_detected(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    (root / "metrics.csv").write_text("value\n999\n", encoding="utf-8")
    report = verify(manifest, root, config, dataset)
    assert report["status"] == "FAIL"
    metrics = next(item for item in report["artifacts"] if item["artifact_id"] == "metrics")
    assert "ARTIFACT_CONTENT_MISMATCH" in metrics["reasons"]


def test_changed_config_dataset_and_revision_are_detected(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    config.write_text('{"window":30}', encoding="utf-8")
    (dataset / "prices.csv").write_text("close\n9.9\n", encoding="utf-8")
    report = verify(manifest, root, config, dataset, revision=OTHER_REV)
    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "FAIL"
    assert "STALE_CONFIG" in codes
    assert "STALE_DATASET" in codes
    assert "STALE_CODE_REVISION" in codes


def test_missing_artifact_is_detected(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    (root / "plan.json").unlink()
    report = verify(manifest, root, config, dataset)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "ARTIFACT_UNAVAILABLE_OR_UNSAFE" for item in report["errors"])


def test_capture_rejects_path_escape_and_symlink(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    with pytest.raises(ValueError, match="artifact_path_outside_root"):
        capture(root, config, dataset, {"artifacts": [{"artifact_id": "escape", "path": "../config.json"}]})
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink_not_allowed"):
        capture(root, config, dataset, {"artifacts": [{"artifact_id": "linked", "path": "linked.txt"}]})


def test_capture_rejects_duplicate_ids_unknown_dependency_and_cycle(tmp_path):
    root, config, dataset, _ = setup_tree(tmp_path)
    with pytest.raises(ValueError, match="duplicate_artifact_id"):
        capture(root, config, dataset, {"artifacts": [
            {"artifact_id": "x", "path": "plan.json"},
            {"artifact_id": "x", "path": "metrics.csv"},
        ]})
    with pytest.raises(ValueError, match="unknown_dependency"):
        capture(root, config, dataset, {"artifacts": [{"artifact_id": "x", "path": "plan.json", "depends_on": ["missing"]}]})
    with pytest.raises(ValueError, match="dependency_cycle"):
        capture(root, config, dataset, {"artifacts": [
            {"artifact_id": "x", "path": "plan.json", "depends_on": ["y"]},
            {"artifact_id": "y", "path": "metrics.csv", "depends_on": ["x"]},
        ]})


def test_capture_rejects_invalid_git_revision(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    with pytest.raises(ValueError, match="source_revision"):
        capture_provenance(spec, source_revision="main", config_path=config, dataset_path=dataset, root=root)


def test_verifier_fails_when_safety_flag_is_not_false(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    manifest["deployment_authorized"] = True
    report = verify(manifest, root, config, dataset)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "safety_flag_not_explicitly_false" for item in report["errors"])


def test_manifest_can_be_serialized_as_stable_json(tmp_path):
    root, config, dataset, spec = setup_tree(tmp_path)
    manifest = capture(root, config, dataset, spec)
    first = json.dumps(manifest, sort_keys=True)
    second = json.dumps(capture(root, config, dataset, spec), sort_keys=True)
    assert first == second

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.research_evidence_pack_builder import assemble_evidence_pack


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def source_reports(tmp_path: Path):
    dataset_hash = digest("dataset")
    registry = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "runs_supplied": 1,
        "runs_validated": 1,
        "run_results": [{"run_id": "run-a", "status": "PASS"}],
        "dataset_fingerprints": {dataset_hash: ["run-a"]},
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
    }
    validation = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "artifacts_checked": 5,
        "missing_required_audits": [],
        "dataset_fingerprints": {dataset_hash: ["run-a"]},
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
    }
    registry_path = tmp_path / "reports" / "registry.json"
    validation_path = tmp_path / "reports" / "validation.json"
    write_json(registry_path, registry)
    write_json(validation_path, validation)
    return registry_path, validation_path


def test_builder_assembles_and_audits_valid_pack(tmp_path):
    registry, validation = source_reports(tmp_path)

    result = assemble_evidence_pack(
        pack_id="test-pack",
        report_specs=[
            f"experiment_registry={registry}",
            f"validation_workflow={validation}",
        ],
        required_roles=None,
        output_path=tmp_path / "pack.json",
    )

    assert result["status"] == "PASS"
    manifest = json.loads((tmp_path / "pack.json").read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "pack_audit.json").read_text(encoding="utf-8"))
    assert manifest["pack_id"] == "test-pack"
    assert audit["status"] == "PASS"
    assert [entry["role"] for entry in manifest["reports"]] == [
        "experiment_registry", "validation_workflow"
    ]


def test_builder_rejects_report_outside_pack_directory(tmp_path):
    inside_registry, inside_validation = source_reports(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    write_json(outside, json.loads(inside_registry.read_text(encoding="utf-8")))
    try:
        with pytest.raises(ValueError, match="report_path_outside_pack_directory"):
            assemble_evidence_pack(
                pack_id="unsafe-pack",
                report_specs=[
                    f"experiment_registry={outside}",
                    f"validation_workflow={inside_validation}",
                ],
                required_roles=None,
                output_path=tmp_path / "pack.json",
            )
    finally:
        outside.unlink(missing_ok=True)


def test_builder_rejects_duplicate_roles_and_does_not_publish(tmp_path):
    registry, validation = source_reports(tmp_path)

    with pytest.raises(ValueError, match="report_roles_must_be_unique"):
        assemble_evidence_pack(
            pack_id="duplicate-role",
            report_specs=[
                f"experiment_registry={registry}",
                f"experiment_registry={validation}",
            ],
            required_roles=None,
            output_path=tmp_path / "pack.json",
        )

    assert not (tmp_path / "pack.json").exists()
    assert not (tmp_path / "pack_audit.json").exists()


def test_builder_never_overwrites_existing_manifest(tmp_path):
    registry, validation = source_reports(tmp_path)
    output = tmp_path / "pack.json"
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        assemble_evidence_pack(
            pack_id="existing-pack",
            report_specs=[
                f"experiment_registry={registry}",
                f"validation_workflow={validation}",
            ],
            required_roles=None,
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_builder_adds_optional_required_roles_without_dropping_core_roles(tmp_path):
    registry_path, validation_path = source_reports(tmp_path)
    dataset_hash = digest("dataset")
    release_path = tmp_path / "reports" / "release.json"
    write_json(release_path, {
        "status": "PASS",
        "readiness": "RELEASE_READY_FOR_HUMAN_REVIEW",
        "errors": [],
        "warnings": [],
        "run_id": "run-a",
        "dataset_fingerprints": {dataset_hash: ["run-a"]},
        "model_qualified": False,
        "deployment_authorized": False,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
    })

    result = assemble_evidence_pack(
        pack_id="release-pack",
        report_specs=[
            f"experiment_registry={registry_path}",
            f"validation_workflow={validation_path}",
            f"release_readiness_run_a={release_path}",
        ],
        required_roles=["release_readiness_run_a"],
        output_path=tmp_path / "release-pack.json",
    )

    assert result["status"] == "PASS"
    manifest = json.loads((tmp_path / "release-pack.json").read_text(encoding="utf-8"))
    assert set(manifest["required_roles"]) == {
        "experiment_registry", "validation_workflow", "release_readiness_run_a"
    }

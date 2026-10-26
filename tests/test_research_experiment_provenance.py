from __future__ import annotations

import hashlib
import json

import pytest

from app.research_experiment_provenance import (
    compare_experiment_manifests,
    validate_experiment_manifest,
    _load_json_object,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(run_id="run-1", *, artifact_path=None, artifact_sha=None):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": "2026-10-25T12:00:00Z",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": sha(b"config-v1"),
        "dataset_sha256": sha(b"dataset-v1"),
        "code_commit_sha": "a" * 40,
        "seed": 11,
        "fold_id": "fold-1",
        "evaluation_window": {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        },
        "metrics": {"net_pnl": 12.5, "max_drawdown": 3.2},
        "artifacts": (
            [{"path": artifact_path, "sha256": artifact_sha}]
            if artifact_path is not None else []
        ),
        "safety_flags": {
            "actionable": False,
            "inferential": False,
            "statistical_significance_claimed": False,
            "automatic_model_selection": False,
            "model_training_performed": False,
            "execution_connected": False,
            "automatic_repair_performed": False,
        },
    }


def test_manifest_passes_with_explicit_provenance_and_safety_flags(tmp_path):
    report = validate_experiment_manifest(manifest(), base_dir=tmp_path)
    assert report["status"] == "PASS"
    assert report["readiness"] == "READY_FOR_DESCRIPTIVE_COMPARISON"
    assert report["actionable"] is False
    assert report["execution_connected"] is False
    assert report["model_training_performed"] is False


def test_artifact_digest_mismatch_fails_closed(tmp_path):
    (tmp_path / "metrics.json").write_text('{"net_pnl": 12}', encoding="utf-8")
    report = validate_experiment_manifest(
        manifest(artifact_path="metrics.json", artifact_sha="0" * 64),
        base_dir=tmp_path,
    )
    assert report["status"] == "FAIL"
    assert any(e["code"] == "artifact_sha256_mismatch" for e in report["errors"])


def test_artifact_path_traversal_fails_closed(tmp_path):
    report = validate_experiment_manifest(
        manifest(artifact_path="../outside.json", artifact_sha=sha(b"x")),
        base_dir=tmp_path,
    )
    assert any(e["code"] == "artifact_path_must_be_confined" for e in report["errors"])


def test_missing_safety_flag_fails_closed(tmp_path):
    payload = manifest()
    del payload["safety_flags"]["execution_connected"]
    report = validate_experiment_manifest(payload, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any("safety_flag_not_explicitly_false:execution_connected" == e["code"] for e in report["errors"])


def test_comparison_is_descriptive_and_does_not_select_winner(tmp_path):
    first = validate_experiment_manifest(manifest("run-1"), base_dir=tmp_path)
    payload = manifest("run-2")
    payload["seed"] = 22
    payload["metrics"] = {"net_pnl": 17.0, "max_drawdown": 4.0}
    second = validate_experiment_manifest(payload, base_dir=tmp_path)
    report = compare_experiment_manifests([first, second])
    assert report["status"] == "PASS"
    assert report["runs_compared"] == 2
    assert report["metric_summaries"]["net_pnl"]["mean"] == 14.75
    assert report["selection_performed"] is False
    assert report["statistical_significance_claimed"] is False


def test_comparison_rejects_dataset_mismatch(tmp_path):
    first = validate_experiment_manifest(manifest("run-1"), base_dir=tmp_path)
    payload = manifest("run-2")
    payload["dataset_sha256"] = sha(b"different dataset")
    second = validate_experiment_manifest(payload, base_dir=tmp_path)
    report = compare_experiment_manifests([first, second])
    assert report["status"] == "FAIL"
    assert any(e["code"] == "comparison_dataset_sha256_mismatch" for e in report["errors"])


def test_comparison_rejects_duplicate_run_ids(tmp_path):
    first = validate_experiment_manifest(manifest("same"), base_dir=tmp_path)
    second = validate_experiment_manifest(manifest("same"), base_dir=tmp_path)
    report = compare_experiment_manifests([first, second])
    assert report["status"] == "FAIL"
    assert any(e["code"] == "duplicate_run_id" for e in report["errors"])


def test_invalid_window_fails_closed(tmp_path):
    payload = manifest()
    payload["evaluation_window"]["end"] = "2025-12-31T00:00:00Z"
    report = validate_experiment_manifest(payload, base_dir=tmp_path)
    assert any(e["code"] == "evaluation_window_order_invalid" for e in report["errors"])


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_strict_json_loader_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="experiment-manifest.json")


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda p: p.update({"run_id": " run-1 "}), "run_id_invalid"),
        (lambda p: p["context"].update({"symbol": " EURUSD "}), "context_symbol_invalid"),
        (lambda p: p.update({"config_sha256": "A" * 64}), "config_sha256_invalid"),
        (lambda p: p.update({"code_commit_sha": "A" * 40}), "code_commit_sha_invalid"),
        (lambda p: p.update({"fold_id": " fold-1 "}), "fold_id_invalid"),
    ],
)
def test_manifest_rejects_noncanonical_identity_values(tmp_path, mutation, error_code):
    payload = manifest()
    mutation(payload)

    report = validate_experiment_manifest(payload, base_dir=tmp_path)

    assert report["status"] == "FAIL"
    assert any(error["code"] == error_code for error in report["errors"])


def test_comparison_rejects_whitespace_padded_run_id_in_pass_report():
    report = compare_experiment_manifests([
        {"status": "PASS", "run_id": " run-a ", "context": {}, "config_sha256": "a", "dataset_sha256": "b", "metrics": {"x": 1}},
        {"status": "PASS", "run_id": "run-b", "context": {}, "config_sha256": "a", "dataset_sha256": "b", "metrics": {"x": 2}},
    ])

    assert report["status"] == "FAIL"
    assert any(error["code"] == "run_id_invalid" for error in report["errors"])

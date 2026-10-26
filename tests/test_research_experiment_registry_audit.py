from __future__ import annotations

import hashlib
import json

import pytest

from app.research_experiment_registry_audit import audit_experiment_registry, _load_json_object


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_manifest(run_id, seed, fold="fold-1", *, dataset=b"dataset-v1"):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": "2026-10-25T12:00:00Z",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": sha(b"config-v1"),
        "dataset_sha256": sha(dataset),
        "code_commit_sha": "a" * 40,
        "seed": seed,
        "fold_id": fold,
        "evaluation_window": {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        },
        "metrics": {"net_pnl": float(seed), "max_drawdown": 2.0},
        "artifacts": [],
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


def write_registry(tmp_path, manifests, **extra):
    runs = []
    for index, payload in enumerate(manifests):
        name = f"run-{index}.json"
        raw = json.dumps(payload, sort_keys=True).encode()
        (tmp_path / name).write_bytes(raw)
        runs.append({"manifest_path": name, "expected_manifest_sha256": sha(raw)})
    return {"schema_version": 1, "registry_id": "test-registry", "runs": runs, **extra}


def test_registry_passes_and_summarizes_seed_runs(tmp_path):
    registry = write_registry(
        tmp_path,
        [run_manifest("run-a", 11), run_manifest("run-b", 22)],
        expected_run_ids=["run-a", "run-b"],
        expected_seeds=[11, 22],
        expected_folds=["fold-1"],
    )
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "PASS"
    assert report["runs_validated"] == 2
    assert report["metric_summaries"]["net_pnl"]["mean"] == 16.5
    assert report["execution_connected"] is False
    assert report["automatic_model_selection"] is False


def test_manifest_checksum_mismatch_fails_closed(tmp_path):
    registry = write_registry(tmp_path, [run_manifest("run-a", 11)])
    registry["runs"][0]["expected_manifest_sha256"] = "0" * 64
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "manifest_sha256_mismatch" for e in report["errors"])


def test_duplicate_run_ids_fail_closed(tmp_path):
    registry = write_registry(
        tmp_path, [run_manifest("duplicate", 11), run_manifest("duplicate", 22)]
    )
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "duplicate_run_id" for e in report["errors"])


def test_dataset_mismatch_fails_closed(tmp_path):
    registry = write_registry(
        tmp_path,
        [run_manifest("run-a", 11), run_manifest("run-b", 22, dataset=b"other dataset")],
    )
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "registry_dataset_sha256_mismatch" for e in report["errors"])


def test_missing_expected_fold_fails_closed(tmp_path):
    registry = write_registry(
        tmp_path,
        [run_manifest("run-a", 11, "fold-1"), run_manifest("run-b", 11, "fold-2")],
        expected_folds=["fold-1", "fold-2", "fold-3"],
    )
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "seed_fold_coverage_incomplete" for e in report["errors"])


def test_path_traversal_fails_closed(tmp_path):
    registry = {
        "schema_version": 1,
        "registry_id": "unsafe",
        "runs": [{"manifest_path": "../outside.json", "expected_manifest_sha256": sha(b"x")}],
    }
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "manifest_path_must_be_confined" for e in report["errors"])


def test_missing_expected_run_fails_closed(tmp_path):
    registry = write_registry(
        tmp_path, [run_manifest("present", 11)], expected_run_ids=["present", "missing"]
    )
    report = audit_experiment_registry(registry, base_dir=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e.get("run_id") == "missing" and e["code"] == "expected_run_missing" for e in report["errors"])


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_strict_json_loader_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="registry.json")

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import research_experiment_registry as registry


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(tmp_path: Path, run_id: str, seed: int, *, timeframe: str = "M15") -> Path:
    path = tmp_path / f"{run_id}_pipeline_manifest.json"
    repro_path = tmp_path / f"{run_id}_reproducibility_audit.json"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "symbol": "EURUSD",
        "timeframe": timeframe,
        "source": "ctrader",
        "source_version": "ctrader-open-api==0.9.2",
        "terms_reference_sha256": sha("https://provider.example/terms"),
        "raw_ohlc_sha256": sha("raw-bars"),
        "dataset_sha256": sha("prepared-dataset"),
        "feature_schema_sha256": sha("feature-schema"),
        "feature_implementation_sha256": sha("feature-code"),
        "label_horizon_bars": 4,
        "class_1_means": "UP",
        "max_label_elapsed_hours": None,
        "label_eligibility_mode": "all_labels_included",
        "raw_ohlc_audit_status": "PASS",
        "feature_source_audit_status": "PASS",
        "point_in_time_feature_invariance_audit_status": "PASS",
        "label_maturity_audit_status": "PASS",
        "provenance_audit_status": "PASS",
        "temporal_calibration_audit_status": "PASS",
        "model_evaluation_gate_status": "PASS",
        "reproducibility_audit_path": str(repro_path),
        "pipeline_manifest_sha256": sha(run_id),
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "research_config": {
            "initial_train_bars": 12000,
            "validation_bars": 1500,
            "test_bars": 1500,
            "step_bars": 1500,
            "trials_per_fold": 2,
            "bootstrap_resamples": 100,
            "max_experiments": 2,
            "include_ablation": False,
            "random_seed": seed,
        },
    }
    write_json(path, manifest)
    write_json(repro_path, {
        "audit_name": "research_experiment_reproducibility_audit",
        "status": "PASS",
        "errors": [],
        "pipeline_manifest_path": str(path),
        "actionable": False,
        "execution_connected": False,
    })
    write_json(tmp_path / f"{run_id}_pipeline_completion_audit.json", {
        "schema_version": 2,
        "run_id": run_id,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "pipeline_manifest_sha256": manifest["pipeline_manifest_sha256"],
    })
    return path


def patch_integrity(monkeypatch, status="PASS"):
    monkeypatch.setattr(registry, "audit_research_run_manifest", lambda path: {
        "status": status, "errors": [], "warnings": []
    })


def test_registry_accepts_matching_protocol_with_distinct_seeds(tmp_path, monkeypatch):
    patch_integrity(monkeypatch)
    first = build_manifest(tmp_path, "run-a", 100)
    second = build_manifest(tmp_path, "run-b", 101)

    report = registry.build_experiment_registry([first, second])

    assert report["status"] == "PASS"
    assert report["runs_supplied"] == 2
    assert report["runs_validated"] == 2
    assert report["random_seeds_distinct"] is True
    assert report["dataset_fingerprints"] == {sha("prepared-dataset"): ["run-a", "run-b"]}
    assert report["actionable"] is False
    assert report["execution_connected"] is False


def test_registry_rejects_protocol_mismatch(tmp_path, monkeypatch):
    patch_integrity(monkeypatch)
    first = build_manifest(tmp_path, "run-a", 100)
    second = build_manifest(tmp_path, "run-b", 101, timeframe="H1")

    report = registry.build_experiment_registry([first, second])

    assert report["status"] == "FAIL"
    assert "timeframe" in report["protocol_mismatches"]
    assert any(item["code"] == "research_protocol_mismatch" for item in report["errors"])


def test_registry_rejects_reused_random_seed(tmp_path, monkeypatch):
    patch_integrity(monkeypatch)
    first = build_manifest(tmp_path, "run-a", 100)
    second = build_manifest(tmp_path, "run-b", 100)

    report = registry.build_experiment_registry([first, second])

    assert report["status"] == "FAIL"
    assert any("random_seed_reused" in error for run in report["run_results"] for error in run["errors"])


def test_registry_rejects_non_clean_cross_artifact_audit(tmp_path, monkeypatch):
    patch_integrity(monkeypatch, status="PASS_WITH_WARNINGS")
    first = build_manifest(tmp_path, "run-a", 100)
    second = build_manifest(tmp_path, "run-b", 101)

    report = registry.build_experiment_registry([first, second])

    assert report["status"] == "FAIL"
    assert all(run["status"] == "FAIL" for run in report["run_results"])


def test_registry_requires_two_runs():
    try:
        registry.build_experiment_registry(["only-one.json"])
    except ValueError as exc:
        assert "at_least_two_manifests_required" in str(exc)
    else:
        raise AssertionError("single-run registry unexpectedly passed")

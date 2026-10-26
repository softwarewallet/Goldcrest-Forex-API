from __future__ import annotations

import hashlib
import json

import pytest

from app.research_readiness_certification_audit import audit_research_readiness


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_report(tmp_path, name, extra=None, status="PASS"):
    payload = {
        "schema_version": 1,
        "audit_name": name,
        "status": status,
        "errors": [],
        "warnings": [],
        "dataset_sha256": sha(b"shared dataset"),
        "actionable": False,
        "inferential": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
    }
    payload.update(extra or {})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload


def standard_reports(tmp_path):
    reports = {}
    reports["feature_temporal_integrity"] = write_report(
        tmp_path,
        "feature_temporal_integrity",
        {
            "prefix_invariance": {"all_passed": True},
            "future_perturbation_invariance": {"all_passed": True},
            "chronological_partition_isolation": {"purge_gaps_sufficient_for_horizon": True},
        },
    )[0]
    reports["feature_backend_parity"] = write_report(
        tmp_path,
        "feature_backend_parity",
        {"comparison_performed": True, "all_features_passed": True},
    )[0]
    reports["pipeline_leakage_boundary"] = write_report(
        tmp_path,
        "pipeline_leakage_boundary",
        {
            "split_results": {"train": {"start_index": 0, "end_index": 69}},
            "preprocessing_results": [{"passed": True}],
            "label_results": [{"passed": True}],
        },
    )[0]
    reports["research_artifact_bundle_certification"] = write_report(
        tmp_path,
        "research_artifact_bundle_certification",
        {"missing_required_audits": [], "artifacts_checked": 3},
    )[0]
    return reports


def manifest_for(reports):
    return {
        "artifacts": [
            {"audit_name": name, "path": path.name}
            for name, path in reports.items()
        ]
    }


def run_audit(tmp_path, manifest):
    return audit_research_readiness(
        manifest,
        manifest_sha256=sha(b"readiness manifest"),
        base_dir=tmp_path,
    )


def test_complete_consistent_bundle_is_ready(tmp_path):
    reports = standard_reports(tmp_path)
    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "PASS"
    assert report["readiness"] == "READY"
    assert report["errors"] == []
    assert report["artifacts_checked"] == 4
    assert report["execution_connected"] is False
    assert report["model_training_performed"] is False
    assert report["actionable"] is False


def test_missing_required_audit_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    reports.pop("pipeline_leakage_boundary")
    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert report["readiness"] == "NOT_READY"
    assert any(error["code"] == "required_audit_missing" for error in report["errors"])


def test_cross_report_dataset_fingerprint_mismatch_fails(tmp_path):
    reports = standard_reports(tmp_path)
    path, payload = write_report(
        tmp_path,
        "feature_backend_parity",
        {
            "comparison_performed": True,
            "all_features_passed": True,
            "dataset_sha256": sha(b"different dataset"),
        },
    )
    reports["feature_backend_parity"] = path
    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "dataset_fingerprint_mismatch_across_reports"
        for error in report["errors"]
    )


def test_non_comparable_backend_does_not_receive_readiness_pass(tmp_path):
    reports = standard_reports(tmp_path)
    path, _ = write_report(
        tmp_path,
        "feature_backend_parity",
        {"comparison_performed": False},
        status="NOT_COMPARABLE",
    )
    reports["feature_backend_parity"] = path
    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(error["code"] == "upstream_audit_not_comparable" for error in report["errors"])


def test_missing_explicit_safety_flag_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["feature_temporal_integrity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["execution_connected"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "required_safety_flag_not_explicitly_false:execution_connected"
        for error in report["errors"]
    )


def test_path_traversal_and_duplicate_audits_are_detected(tmp_path):
    reports = standard_reports(tmp_path)
    manifest = manifest_for(reports)
    manifest["artifacts"].append({
        "audit_name": "feature_temporal_integrity",
        "path": "../outside.json",
    })
    report = run_audit(tmp_path, manifest)

    codes = {error["code"] for error in report["errors"]}
    assert "duplicate_audit_name" in codes
    assert "artifact_path_must_be_relative_and_confined" in codes


def test_manifest_validation_rejects_duplicate_required_names(tmp_path):
    reports = standard_reports(tmp_path)
    manifest = manifest_for(reports)
    manifest["required_audits"] = ["feature_temporal_integrity", "feature_temporal_integrity"]

    with pytest.raises(ValueError, match="required_audits_must_be_unique"):
        run_audit(tmp_path, manifest)


def test_missing_optional_pipeline_evidence_warns_not_passes_silently(tmp_path):
    reports = standard_reports(tmp_path)
    path, _ = write_report(
        tmp_path,
        "pipeline_leakage_boundary",
        {"split_results": {"train": {"start_index": 0, "end_index": 69}}},
    )
    reports["pipeline_leakage_boundary"] = path
    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["readiness"] == "READY_WITH_WARNINGS"
    assert "pipeline_preprocessing_evidence_missing" in report["warnings"]
    assert "pipeline_label_evidence_missing" in report["warnings"]

@pytest.mark.parametrize("raw", [b'{"metric":NaN}', b'{"metric":Infinity}', b'{"metric":-Infinity}', b'{"value":1,"value":2}'])
def test_certification_json_readers_reject_nonstandard_json(tmp_path, raw):
    from app.research_readiness_certification_audit import _load_json_object
    with pytest.raises(ValueError):
        _load_json_object(raw, source="fixture.json")

def test_missing_errors_array_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["feature_temporal_integrity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("errors")
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "upstream_errors_field_missing_or_invalid"
        and error["audit_name"] == "feature_temporal_integrity"
        for error in report["errors"]
    )


def test_malformed_warnings_array_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["feature_backend_parity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["warnings"] = {"not": "a-list"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "upstream_warnings_field_invalid"
        and error["audit_name"] == "feature_backend_parity"
        for error in report["errors"]
    )

def test_explicit_null_warnings_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["feature_backend_parity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["warnings"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_audit(tmp_path, manifest_for(reports))
    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "upstream_warnings_field_invalid"
        and error["audit_name"] == "feature_backend_parity"
        for error in report["errors"]
    )



@pytest.mark.parametrize(
    ("status", "warnings"),
    [("PASS", ["warning not reflected in status"]), ("PASS_WITH_WARNINGS", [])],
)
def test_readiness_rejects_status_warning_mismatch(tmp_path, status, warnings):
    reports = standard_reports(tmp_path)
    path, payload = write_report(
        tmp_path,
        "feature_backend_parity",
        {"comparison_performed": True, "all_features_passed": True},
        status=status,
    )
    payload["warnings"] = warnings
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    reports["feature_backend_parity"] = path

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert "status_warnings_mismatch" in str(report)


@pytest.mark.parametrize("status", [["PASS"], {"status": "PASS"}, None, 7])
def test_readiness_malformed_upstream_status_fails_closed_without_raising(tmp_path, status):
    reports = standard_reports(tmp_path)
    path = reports["feature_backend_parity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_audit(tmp_path, manifest_for(reports))
    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "status_unknown_or_missing"
        and error["audit_name"] == "feature_backend_parity"
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix_invariance", []),
        ("future_perturbation_invariance", {"all_passed": []}),
        ("chronological_partition_isolation", None),
    ],
)
def test_malformed_temporal_evidence_fails_closed_without_raising(tmp_path, field, value):
    reports = standard_reports(tmp_path)
    path = reports["feature_temporal_integrity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_audit(tmp_path, manifest_for(reports))
    assert report["status"] == "FAIL"
    assert report["readiness"] == "NOT_READY"


def test_readiness_missing_warnings_collection_fails_closed(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["feature_backend_parity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("warnings")
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_audit(tmp_path, manifest_for(reports))
    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "upstream_warnings_field_invalid"
        and error["audit_name"] == "feature_backend_parity"
        for error in report["errors"]
    )



def test_readiness_rejects_boolean_bundle_artifact_count(tmp_path):
    reports = standard_reports(tmp_path)
    path = reports["research_artifact_bundle_certification"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts_checked"] = True
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "bundle_artifact_count_invalid"
        and error["audit_name"] == "research_artifact_bundle_certification"
        for error in report["errors"]
    )



@pytest.mark.parametrize("schema_version", [True, False, 1.0, "1", None, 2])
def test_readiness_rejects_invalid_upstream_schema_version(tmp_path, schema_version):
    reports = standard_reports(tmp_path)
    path = reports["feature_backend_parity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = run_audit(tmp_path, manifest_for(reports))

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "schema_version_invalid"
        and error["audit_name"] == "feature_backend_parity"
        for error in report["errors"]
    )

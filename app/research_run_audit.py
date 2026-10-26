"""Cross-artifact integrity audit for completed offline research runs.

This checks manifest fingerprints and consistency between generated artifacts. It does not
recompute model metrics, establish absence of leakage, or validate trading profitability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


_ARTIFACT_FIELDS = {
    "raw_ohlc_path": "raw_ohlc_sha256",
    "dataset_path": "dataset_sha256",
    "dataset_manifest_path": "dataset_manifest_sha256",
    "experiment_matrix_report": "experiment_matrix_report_sha256",
    "provenance_audit_path": "provenance_audit_sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact_path(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        relative_to_manifest = manifest_path.parent / candidate
        if relative_to_manifest.is_file():
            return relative_to_manifest
    return candidate


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def audit_research_run_manifest(manifest_file: str | Path) -> dict[str, Any]:
    """Audit a pipeline manifest and all artifacts it references."""
    manifest_path = Path(manifest_file)
    errors: list[str] = []
    warnings: list[str] = []
    checked_artifacts: dict[str, dict[str, str]] = {}
    related: dict[str, Mapping[str, Any]] = {}

    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "manifest_path": str(manifest_path),
            "errors": [f"manifest_unreadable:{type(exc).__name__}"],
            "warnings": [],
            "checked_artifacts": {},
            "actionable": False,
            "execution_connected": False,
        }

    if manifest.get("schema_version") != 1:
        errors.append("unsupported_or_missing_schema_version")
    if not isinstance(manifest.get("run_id"), str) or not manifest.get("run_id", "").strip():
        errors.append("missing_run_id")
    for field in ("actionable", "execution_connected", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if manifest.get(field) is not False:
            errors.append(f"{field}_must_be_false")

    expected_manifest_hash = manifest.get("pipeline_manifest_sha256")
    hash_payload = dict(manifest)
    hash_payload.pop("pipeline_manifest_sha256", None)
    try:
        canonical = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual_manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if expected_manifest_hash != actual_manifest_hash:
            errors.append("pipeline_manifest_sha256_mismatch")
    except (TypeError, ValueError):
        errors.append("pipeline_manifest_not_canonical_json")

    for path_field, hash_field in _ARTIFACT_FIELDS.items():
        resolved = _resolve_artifact_path(manifest.get(path_field), manifest_path)
        expected_hash = manifest.get(hash_field)
        if resolved is None:
            errors.append(f"{path_field}_missing")
            continue
        if not resolved.is_file():
            errors.append(f"{path_field}_not_found")
            continue
        actual_hash = _sha256(resolved)
        checked_artifacts[path_field] = {
            "path": str(resolved),
            "sha256": actual_hash,
        }
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append(f"{path_field}_sha256_mismatch")
        if path_field in {"dataset_manifest_path", "experiment_matrix_report", "provenance_audit_path"}:
            try:
                related[path_field] = _read_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path_field}_invalid_json:{type(exc).__name__}")

    # Newer pipeline runs include a raw-data audit receipt. Keep old schema-v1
    # manifests readable, but fully verify the receipt whenever it is present.
    if manifest.get("raw_ohlc_audit_path") or manifest.get("raw_ohlc_audit_sha256"):
        resolved = _resolve_artifact_path(manifest.get("raw_ohlc_audit_path"), manifest_path)
        expected_hash = manifest.get("raw_ohlc_audit_sha256")
        if resolved is None:
            errors.append("raw_ohlc_audit_path_missing")
        elif not resolved.is_file():
            errors.append("raw_ohlc_audit_path_not_found")
        else:
            actual_hash = _sha256(resolved)
            checked_artifacts["raw_ohlc_audit_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append("raw_ohlc_audit_path_sha256_mismatch")
            try:
                related["raw_ohlc_audit_path"] = _read_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"raw_ohlc_audit_path_invalid_json:{type(exc).__name__}")

    # Newer runs also include a calibration-stability receipt derived from the
    # persisted experiment matrix. Keep older schema-v1 manifests readable.
    if manifest.get("temporal_calibration_audit_path") or manifest.get("temporal_calibration_audit_sha256"):
        resolved = _resolve_artifact_path(manifest.get("temporal_calibration_audit_path"), manifest_path)
        expected_hash = manifest.get("temporal_calibration_audit_sha256")
        if resolved is None:
            errors.append("temporal_calibration_audit_path_missing")
        elif not resolved.is_file():
            errors.append("temporal_calibration_audit_path_not_found")
        else:
            actual_hash = _sha256(resolved)
            checked_artifacts["temporal_calibration_audit_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append("temporal_calibration_audit_path_sha256_mismatch")
            try:
                related["temporal_calibration_audit_path"] = _read_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"temporal_calibration_audit_path_invalid_json:{type(exc).__name__}")

    temporal_audit = related.get("temporal_calibration_audit_path", {})
    if temporal_audit:
        matrix_hash = checked_artifacts.get("experiment_matrix_report", {}).get("sha256")
        if temporal_audit.get("scope") != "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT":
            errors.append("temporal_calibration_audit_scope_invalid")
        if temporal_audit.get("run_id") != manifest.get("run_id"):
            errors.append("temporal_calibration_audit_run_id_mismatch")
        if temporal_audit.get("experiment_matrix_sha256") != matrix_hash:
            errors.append("temporal_calibration_audit_source_matrix_hash_mismatch")
        matrix_report = related.get("experiment_matrix_report", {})
        claimed_payload_hash = matrix_report.get("report_payload_sha256")
        payload_hash_verified = False
        if isinstance(claimed_payload_hash, str):
            canonical_payload = dict(matrix_report)
            canonical_payload.pop("report_payload_sha256", None)
            try:
                canonical_matrix = json.dumps(
                    canonical_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                payload_hash_verified = (
                    hashlib.sha256(canonical_matrix.encode("utf-8")).hexdigest()
                    == claimed_payload_hash
                )
            except (TypeError, ValueError):
                payload_hash_verified = False
        if not payload_hash_verified:
            errors.append("temporal_calibration_audit_source_matrix_payload_hash_invalid")
        if temporal_audit.get("experiment_matrix_payload_hash_verified") is not True:
            errors.append("temporal_calibration_audit_matrix_payload_hash_unverified")
        for field in ("actionable", "execution_connected", "automatic_model_selection", "test_metrics_used_for_selection"):
            if temporal_audit.get(field) is not False:
                errors.append(f"temporal_calibration_audit_{field}_must_be_false")
        if temporal_audit.get("status") != manifest.get("temporal_calibration_audit_status"):
            errors.append("temporal_calibration_audit_status_mismatch")
        if temporal_audit.get("status") == "PASS_WITH_WARNINGS":
            warnings.append("temporal_calibration_audit_passed_with_warnings")
        elif temporal_audit.get("status") != "PASS":
            errors.append("temporal_calibration_audit_not_pass")
        comparisons = temporal_audit.get("comparisons")
        if (
            not isinstance(comparisons, list)
            or temporal_audit.get("comparison_count") != len(comparisons)
        ):
            errors.append("temporal_calibration_audit_comparison_count_mismatch")

    raw_audit = related.get("raw_ohlc_audit_path", {})
    raw_hash = checked_artifacts.get("raw_ohlc_path", {}).get("sha256")
    if "raw_ohlc_audit_path" in related:
        if raw_audit.get("snapshot_sha256") != raw_hash:
            errors.append("raw_ohlc_audit_snapshot_sha256_mismatch")
        if raw_audit.get("status") != manifest.get("raw_ohlc_audit_status"):
            errors.append("raw_ohlc_audit_status_mismatch")
        if raw_audit.get("errors") != []:
            errors.append("raw_ohlc_audit_contains_errors")
        if raw_audit.get("actionable") is not False or raw_audit.get("execution_connected") is not False:
            errors.append("raw_ohlc_audit_safety_flags_invalid")
        if raw_audit.get("status") == "PASS_WITH_WARNINGS":
            warnings.append("raw_ohlc_audit_passed_with_warnings")
        elif raw_audit.get("status") != "PASS":
            errors.append("raw_ohlc_audit_not_pass")

    dataset_hash = checked_artifacts.get("dataset_path", {}).get("sha256")
    dataset_manifest = related.get("dataset_manifest_path", {})
    if dataset_hash and dataset_manifest.get("output_dataset_sha256") != dataset_hash:
        errors.append("dataset_manifest_output_hash_mismatch")

    label_cap = manifest.get("max_label_elapsed_hours")
    eligibility_mode = manifest.get("label_eligibility_mode", "all_labels_included")
    dataset_eligibility = dataset_manifest.get("label_eligibility")
    if eligibility_mode == "elapsed_time_cap":
        if (
            isinstance(label_cap, bool)
            or not isinstance(label_cap, (int, float))
            or label_cap <= 0
        ):
            errors.append("pipeline_label_eligibility_cap_invalid")
        if not isinstance(dataset_eligibility, dict):
            errors.append("dataset_manifest_label_eligibility_missing")
        elif (
            dataset_eligibility.get("mode") != "elapsed_time_cap"
            or dataset_eligibility.get("max_label_elapsed_hours") != label_cap
            or dataset_eligibility.get("all_rows_preserved_for_fixed_temporal_folds") is not True
            or dataset_eligibility.get("labels_rewritten") is not False
        ):
            errors.append("dataset_manifest_label_eligibility_mismatch")
    elif label_cap is not None:
        errors.append("pipeline_label_eligibility_mode_cap_mismatch")
    elif isinstance(dataset_eligibility, dict):
        errors.append("dataset_manifest_label_eligibility_cap_not_in_pipeline_manifest")

    report = related.get("experiment_matrix_report", {})
    if report:
        if report.get("run_id") != manifest.get("run_id"):
            errors.append("experiment_report_run_id_mismatch")
        if report.get("dataset_sha256") != dataset_hash:
            errors.append("experiment_report_dataset_sha256_mismatch")
        if report.get("label_eligibility_mode", "all_labels_included") != eligibility_mode:
            errors.append("experiment_report_label_eligibility_mode_mismatch")
        if report.get("max_label_elapsed_hours") != label_cap:
            errors.append("experiment_report_label_eligibility_cap_mismatch")
        for field in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
            if report.get(field) is not False:
                errors.append(f"experiment_report_{field}_must_be_false")

    audit = related.get("provenance_audit_path", {})
    report_hash = checked_artifacts.get("experiment_matrix_report", {}).get("sha256")
    if audit:
        if audit.get("source_report_sha256") != report_hash:
            errors.append("provenance_audit_source_report_hash_mismatch")
        if audit.get("run_id") != manifest.get("run_id"):
            errors.append("provenance_audit_run_id_mismatch")
        if audit.get("status") != manifest.get("provenance_audit_status"):
            errors.append("provenance_audit_status_mismatch")
        if audit.get("actionable") is not False or audit.get("execution_connected") is not False:
            errors.append("provenance_audit_safety_flags_invalid")
        if audit.get("status") == "PASS_WITH_WARNINGS":
            warnings.append("provenance_audit_passed_with_warnings")
        elif audit.get("status") != "PASS":
            errors.append("provenance_audit_not_pass")

    # Newer runs emit a final completion receipt after the manifest and validation
    # report have been persisted. Verify its identity and fail-closed integrity findings
    # when present, while keeping older schema-v1 runs readable.
    completion_path = manifest_path.parent / f"{manifest.get('run_id', '')}_pipeline_completion_audit.json"
    if completion_path.is_file():
        try:
            completion = _read_json(completion_path)
            checked_artifacts["pipeline_completion_audit_path"] = {
                "path": str(completion_path.resolve()),
                "sha256": _sha256(completion_path),
            }
            if completion.get("audit_name") != "research_pipeline_completion_audit":
                errors.append("pipeline_completion_audit_name_mismatch")
            if completion.get("run_id") != manifest.get("run_id"):
                errors.append("pipeline_completion_audit_run_id_mismatch")
            if completion.get("pipeline_manifest_sha256") != manifest.get("pipeline_manifest_sha256"):
                errors.append("pipeline_completion_audit_manifest_hash_mismatch")
            if completion.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
                errors.append("pipeline_completion_audit_not_passing")
            if completion.get("errors") != []:
                errors.append("pipeline_completion_audit_contains_errors")
            for field in ("actionable", "execution_connected", "model_qualified", "deployment_authorized"):
                if completion.get(field) is not False:
                    errors.append(f"pipeline_completion_audit_{field}_must_be_false")
            checks = completion.get("checks")
            if not isinstance(checks, dict):
                errors.append("pipeline_completion_audit_checks_invalid")
            else:
                for field in (
                    "pipeline_manifest_self_hash",
                    "reproducibility_audit_status",
                    "model_evaluation_gate_file_hash",
                ):
                    if checks.get(field) != "PASS":
                        errors.append(f"pipeline_completion_audit_{field}_not_pass")
            if completion.get("status") == "PASS_WITH_WARNINGS":
                warnings.append("pipeline_completion_audit_passed_with_warnings")
            if completion.get("warnings") and not isinstance(completion.get("warnings"), list):
                errors.append("pipeline_completion_audit_warnings_invalid")

            # Schema v2 records the exact source-report bytes observed when the completion
            # receipt was produced. Re-check those hashes so later edits to validation or
            # reproducibility receipts cannot pass merely because their run IDs still match.
            if completion.get("schema_version") == 2:
                validation_path = manifest_path.parent / f"{manifest.get('run_id', '')}_validation_report.json"
                reproducibility_path = _resolve_artifact_path(
                    manifest.get("reproducibility_audit_path"), manifest_path
                )
                evaluation_path = _resolve_artifact_path(
                    manifest.get("model_evaluation_gate_path"), manifest_path
                )
                source_specs = (
                    ("validation_report_sha256", validation_path, "validation_report"),
                    ("reproducibility_audit_sha256", reproducibility_path, "reproducibility_audit"),
                    ("model_evaluation_gate_sha256", evaluation_path, "model_evaluation_gate"),
                )
                for hash_field, source_path, label in source_specs:
                    expected_hash = completion.get(hash_field)
                    if source_path is None or not source_path.is_file():
                        errors.append(f"pipeline_completion_audit_{label}_source_missing")
                        continue
                    try:
                        actual_hash = _sha256(source_path)
                    except OSError:
                        errors.append(f"pipeline_completion_audit_{label}_source_unreadable")
                        continue
                    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                        errors.append(f"pipeline_completion_audit_{label}_source_hash_mismatch")
            elif completion.get("schema_version") == 1:
                # Existing receipts remain readable, but they predate source-byte binding.
                warnings.append("pipeline_completion_audit_source_hashes_unavailable")
            else:
                errors.append("pipeline_completion_audit_schema_version_unsupported")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"pipeline_completion_audit_unreadable:{type(exc).__name__}")

    if manifest.get("provenance_audit_status") == "FAIL":
        errors.append("pipeline_provenance_audit_failed")
    if not checked_artifacts:
        warnings.append("no_artifacts_were_verified")

    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "checked_artifact_count": len(checked_artifacts),
        "checked_artifacts": checked_artifacts,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Cross-artifact integrity and recorded-safeguard checks only. This audit does not "
            "recompute metrics, prove absence of leakage, validate data rights, or establish profitability."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify hashes and consistency for a completed Goldcrest offline research run."
    )
    parser.add_argument("manifest", help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    result = audit_research_run_manifest(args.manifest)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Research run audit: {result['status']}")
        print(f"Run ID: {result.get('run_id', 'unknown')}")
        print(f"Artifacts checked: {result.get('checked_artifact_count', 0)}")
        for error in result.get("errors", []):
            print(f"ERROR: {error}")
        for warning in result.get("warnings", []):
            print(f"WARNING: {warning}")
        print("Actionable: false")
    return 0 if result["status"] != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())

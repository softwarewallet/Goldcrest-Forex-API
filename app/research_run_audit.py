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

from app.research_experiment_fingerprint import canonical_json, verify_experiment_fingerprint


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


def _read_json(path: Path, *, raw_bytes: bytes | None = None) -> Mapping[str, Any]:
    raw = path.read_bytes() if raw_bytes is None else raw_bytes
    value = json.loads(raw.decode("utf-8"))
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
    # Bind every digest and parsed JSON object to the same byte snapshot.
    # Recheck all observed files at the end to detect mutation during the audit.
    snapshots: dict[Path, bytes] = {}

    def snapshot_bytes(path: Path) -> bytes:
        key = Path(path).resolve()
        if key not in snapshots:
            snapshots[key] = key.read_bytes()
        return snapshots[key]

    def snapshot_hash(path: Path) -> str:
        return hashlib.sha256(snapshot_bytes(path)).hexdigest()

    def snapshot_json(path: Path) -> Mapping[str, Any]:
        return _read_json(path, raw_bytes=snapshot_bytes(path))

    try:
        manifest = snapshot_json(manifest_path)
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
        actual_hash = snapshot_hash(resolved)
        checked_artifacts[path_field] = {
            "path": str(resolved),
            "sha256": actual_hash,
        }
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append(f"{path_field}_sha256_mismatch")
        if path_field in {"dataset_manifest_path", "experiment_matrix_report", "provenance_audit_path"}:
            try:
                related[path_field] = snapshot_json(resolved)
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
            actual_hash = snapshot_hash(resolved)
            checked_artifacts["raw_ohlc_audit_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append("raw_ohlc_audit_path_sha256_mismatch")
            try:
                related["raw_ohlc_audit_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"raw_ohlc_audit_path_invalid_json:{type(exc).__name__}")

    # Newer runs include a point-in-time feature invariance receipt. Verify its
    # content hash, identity, source snapshot binding, and non-actionable safety flags.
    if (
        manifest.get("point_in_time_feature_invariance_audit_path")
        or manifest.get("point_in_time_feature_invariance_audit_sha256")
    ):
        field = "point_in_time_feature_invariance_audit"
        resolved = _resolve_artifact_path(
            manifest.get(f"{field}_path"), manifest_path
        )
        expected_hash = manifest.get(f"{field}_sha256")
        if resolved is None:
            errors.append(f"{field}_path_missing")
        elif not resolved.is_file():
            errors.append(f"{field}_path_not_found")
        else:
            actual_hash = snapshot_hash(resolved)
            checked_artifacts[f"{field}_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append(f"{field}_path_sha256_mismatch")
            try:
                related[f"{field}_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{field}_path_invalid_json:{type(exc).__name__}")

        point_in_time = related.get(f"{field}_path", {})
        if point_in_time:
            if point_in_time.get("audit_name") != "research_point_in_time_feature_invariance_audit":
                errors.append(f"{field}_audit_name_mismatch")
            if point_in_time.get("status") != manifest.get(f"{field}_status"):
                errors.append(f"{field}_status_mismatch")
            if point_in_time.get("source_sha256") != checked_artifacts.get("raw_ohlc_path", {}).get("sha256"):
                errors.append(f"{field}_source_ohlc_hash_mismatch")
            for flag in ("actionable", "execution_connected", "model_training_performed", "automatic_model_selection"):
                if point_in_time.get(flag) is not False:
                    errors.append(f"{field}_{flag}_must_be_false")
            if point_in_time.get("status") != "PASS" or point_in_time.get("errors") != []:
                errors.append(f"{field}_not_pass")
            if point_in_time.get("status") == "PASS" and point_in_time.get("cutoff_count", 0) < 1:
                errors.append(f"{field}_cutoff_coverage_missing")

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
            actual_hash = snapshot_hash(resolved)
            checked_artifacts["temporal_calibration_audit_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append("temporal_calibration_audit_path_sha256_mismatch")
            try:
                related["temporal_calibration_audit_path"] = snapshot_json(resolved)
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

    # Verify the immutable experiment fingerprint receipt when present. This binds
    # the configuration/code/data identity to the exact raw, dataset, and manifest bytes.
    if manifest.get("experiment_fingerprint_path") or manifest.get("experiment_fingerprint_file_sha256"):
        field = "experiment_fingerprint"
        resolved = _resolve_artifact_path(manifest.get(f"{field}_path"), manifest_path)
        expected_file_hash = manifest.get(f"{field}_file_sha256")
        if resolved is None:
            errors.append(f"{field}_path_missing")
        elif not resolved.is_file():
            errors.append(f"{field}_path_not_found")
        else:
            actual_file_hash = snapshot_hash(resolved)
            checked_artifacts[f"{field}_path"] = {
                "path": str(resolved),
                "sha256": actual_file_hash,
            }
            if not isinstance(expected_file_hash, str) or actual_file_hash != expected_file_hash:
                errors.append(f"{field}_file_sha256_mismatch")
            try:
                related[f"{field}_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{field}_path_invalid_json:{type(exc).__name__}")

        fingerprint = related.get(f"{field}_path", {})
        if fingerprint:
            verification = verify_experiment_fingerprint(dict(fingerprint))
            if verification.get("status") != "PASS":
                errors.append(f"{field}_self_integrity_failed")
            if fingerprint.get("experiment_fingerprint_sha256") != manifest.get("experiment_fingerprint_sha256"):
                errors.append(f"{field}_hash_link_mismatch")
            if fingerprint.get("raw_ohlc_sha256") != checked_artifacts.get("raw_ohlc_path", {}).get("sha256"):
                errors.append(f"{field}_raw_ohlc_hash_mismatch")
            if fingerprint.get("dataset_sha256") != checked_artifacts.get("dataset_path", {}).get("sha256"):
                errors.append(f"{field}_dataset_hash_mismatch")
            dataset_manifest_path = checked_artifacts.get("dataset_manifest_path", {}).get("path")
            dataset_manifest_file_hash = checked_artifacts.get("dataset_manifest_path", {}).get("sha256")
            if fingerprint.get("dataset_manifest_file_sha256") != dataset_manifest_file_hash:
                errors.append(f"{field}_dataset_manifest_file_hash_mismatch")
            if dataset_manifest_path:
                try:
                    dataset_manifest_payload = snapshot_json(Path(dataset_manifest_path))
                    unhashed_dataset_manifest = dict(dataset_manifest_payload)
                    unhashed_dataset_manifest.pop("manifest_sha256", None)
                    actual_content_hash = hashlib.sha256(canonical_json(unhashed_dataset_manifest)).hexdigest()
                    if fingerprint.get("dataset_manifest_content_sha256") != actual_content_hash:
                        errors.append(f"{field}_dataset_manifest_content_hash_mismatch")
                    if dataset_manifest_payload.get("feature_schema_sha256") != fingerprint.get("feature_schema_sha256"):
                        errors.append(f"{field}_feature_schema_hash_mismatch")
                    if dataset_manifest_payload.get("feature_implementation_sha256") != fingerprint.get("feature_implementation_sha256"):
                        errors.append(f"{field}_feature_implementation_hash_mismatch")
                except (OSError, json.JSONDecodeError, ValueError):
                    errors.append(f"{field}_dataset_manifest_unreadable")
            for flag in ("actionable", "execution_connected", "automatic_model_selection"):
                if fingerprint.get(flag) is not False:
                    errors.append(f"{field}_{flag}_must_be_false")

    # Verify the reproducibility receipt directly, even when no completion receipt
    # exists. It must be a clean PASS and refer to this manifest and (when present) its
    # exact experiment fingerprint file.
    if manifest.get("reproducibility_audit_path") or manifest.get("reproducibility_audit_sha256"):
        field = "reproducibility_audit"
        resolved = _resolve_artifact_path(manifest.get(f"{field}_path"), manifest_path)
        if resolved is None:
            errors.append(f"{field}_path_missing")
        elif not resolved.is_file():
            errors.append(f"{field}_path_not_found")
        else:
            actual_hash = snapshot_hash(resolved)
            checked_artifacts[f"{field}_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            try:
                related[f"{field}_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{field}_path_invalid_json:{type(exc).__name__}")

        reproducibility = related.get(f"{field}_path", {})
        if reproducibility:
            if reproducibility.get("audit_name") != "research_experiment_reproducibility_audit":
                errors.append(f"{field}_audit_name_mismatch")
            if reproducibility.get("status") != "PASS" or reproducibility.get("errors") != []:
                errors.append(f"{field}_not_pass")
            if reproducibility.get("actionable") is not False:
                errors.append(f"{field}_actionable_must_be_false")
            for flag in ("execution_connected", "automatic_model_selection"):
                if reproducibility.get(flag) is not False:
                    errors.append(f"{field}_{flag}_must_be_false")
            recorded_manifest_path = reproducibility.get("pipeline_manifest_path")
            if not isinstance(recorded_manifest_path, str) or Path(recorded_manifest_path).expanduser().resolve() != manifest_path.expanduser().resolve():
                errors.append(f"{field}_pipeline_manifest_path_mismatch")

            fingerprint_path_value = manifest.get("experiment_fingerprint_path")
            if fingerprint_path_value:
                expected_fingerprint_path = _resolve_artifact_path(fingerprint_path_value, manifest_path)
                recorded_fingerprint_path = reproducibility.get("fingerprint_path")
                if (
                    expected_fingerprint_path is None
                    or not isinstance(recorded_fingerprint_path, str)
                    or Path(recorded_fingerprint_path).expanduser().resolve() != expected_fingerprint_path.resolve()
                ):
                    errors.append(f"{field}_fingerprint_path_mismatch")
                expected_fingerprint_hash = checked_artifacts.get("experiment_fingerprint_path", {}).get("sha256")
                if reproducibility.get("fingerprint_file_sha256") != expected_fingerprint_hash:
                    errors.append(f"{field}_fingerprint_file_hash_mismatch")

    # Verify the persisted model-evaluation gate receipt when the manifest declares it.
    # The gate must be tied to the exact experiment-matrix bytes checked above.
    if manifest.get("model_evaluation_gate_path") or manifest.get("model_evaluation_gate_sha256"):
        field = "model_evaluation_gate"
        resolved = _resolve_artifact_path(manifest.get(f"{field}_path"), manifest_path)
        expected_hash = manifest.get(f"{field}_sha256")
        if resolved is None:
            errors.append(f"{field}_path_missing")
        elif not resolved.is_file():
            errors.append(f"{field}_path_not_found")
        else:
            actual_hash = snapshot_hash(resolved)
            checked_artifacts[f"{field}_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append(f"{field}_path_sha256_mismatch")
            try:
                related[f"{field}_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{field}_path_invalid_json:{type(exc).__name__}")

        evaluation = related.get(f"{field}_path", {})
        if evaluation:
            if evaluation.get("scope") != "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE":
                errors.append(f"{field}_scope_invalid")
            if evaluation.get("run_id") != manifest.get("run_id"):
                errors.append(f"{field}_run_id_mismatch")
            matrix_hash = checked_artifacts.get("experiment_matrix_report", {}).get("sha256")
            if evaluation.get("source_matrix_sha256") != matrix_hash:
                errors.append(f"{field}_source_matrix_hash_mismatch")
            if evaluation.get("status") != manifest.get(f"{field}_status"):
                errors.append(f"{field}_status_mismatch")
            if evaluation.get("status") != "PASS" or evaluation.get("errors") != []:
                errors.append(f"{field}_not_pass")
            for identity_field in ("symbol", "timeframe"):
                manifest_identity = manifest.get(identity_field)
                if manifest_identity is not None and evaluation.get(identity_field) != manifest_identity:
                    errors.append(f"{field}_{identity_field}_mismatch")
            matrix_report = related.get("experiment_matrix_report", {})
            matrix_experiments = matrix_report.get("experiments")
            if isinstance(matrix_experiments, list):
                if evaluation.get("experiment_count") != len(matrix_experiments):
                    errors.append(f"{field}_experiment_count_mismatch")
                expected_folds = 0
                expected_metric_objects = 0
                for experiment in matrix_experiments:
                    if not isinstance(experiment, dict):
                        continue
                    fold_results = experiment.get("fold_results")
                    if not isinstance(fold_results, list):
                        continue
                    expected_folds += len(fold_results)
                    expected_metric_objects += sum(
                        1 for fold in fold_results
                        if isinstance(fold, dict) and isinstance(fold.get("test_metrics"), dict)
                    )
                if evaluation.get("folds_checked") != expected_folds:
                    errors.append(f"{field}_fold_count_mismatch")
                if evaluation.get("fold_metric_objects_checked") != expected_metric_objects:
                    errors.append(f"{field}_fold_metric_count_mismatch")
            for flag in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
                if evaluation.get(flag) is not False:
                    errors.append(f"{field}_{flag}_must_be_false")

    # Newer pipeline runs include a feature-source provenance receipt. Verify its
    # exact file hash, dataset binding, dataset-manifest fingerprints, and safety flags.
    # Legacy manifests without this receipt remain readable.
    if manifest.get("feature_source_audit_path") or manifest.get("feature_source_audit_sha256"):
        field = "feature_source_audit"
        resolved = _resolve_artifact_path(manifest.get(f"{field}_path"), manifest_path)
        expected_hash = manifest.get(f"{field}_sha256")
        if resolved is None:
            errors.append(f"{field}_path_missing")
        elif not resolved.is_file():
            errors.append(f"{field}_path_not_found")
        else:
            actual_hash = snapshot_hash(resolved)
            checked_artifacts[f"{field}_path"] = {
                "path": str(resolved),
                "sha256": actual_hash,
            }
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                errors.append(f"{field}_path_sha256_mismatch")
            try:
                related[f"{field}_path"] = snapshot_json(resolved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{field}_path_invalid_json:{type(exc).__name__}")

        feature_audit = related.get(f"{field}_path", {})
        if feature_audit:
            if feature_audit.get("audit_name") != "research_feature_source_provenance_audit":
                errors.append(f"{field}_audit_name_mismatch")
            if feature_audit.get("status") != manifest.get(f"{field}_status"):
                errors.append(f"{field}_status_mismatch")
            if feature_audit.get("dataset_sha256") != dataset_hash:
                errors.append(f"{field}_dataset_sha256_mismatch")
            if feature_audit.get("errors") != []:
                errors.append(f"{field}_contains_errors")
            if feature_audit.get("status") != "PASS":
                errors.append(f"{field}_not_pass")
            if feature_audit.get("actionable") is not False:
                errors.append(f"{field}_actionable_must_be_false")
            for flag in ("execution_connected", "model_training_performed", "automatic_model_selection"):
                if feature_audit.get(flag) is not False:
                    errors.append(f"{field}_{flag}_must_be_false")

            declared_manifest_hash = dataset_manifest.get("manifest_sha256")
            if declared_manifest_hash is not None and feature_audit.get("manifest_sha256_declared") != declared_manifest_hash:
                errors.append(f"{field}_dataset_manifest_hash_mismatch")
            declared_schema_hash = manifest.get("feature_schema_sha256")
            if declared_schema_hash is not None and feature_audit.get("feature_schema_sha256_declared") != declared_schema_hash:
                errors.append(f"{field}_feature_schema_hash_mismatch")
            declared_implementation_hash = manifest.get("feature_implementation_sha256")
            if declared_implementation_hash is not None and feature_audit.get("feature_implementation_sha256_declared") != declared_implementation_hash:
                errors.append(f"{field}_feature_implementation_hash_mismatch")
            declared_feature_count = manifest.get("feature_count")
            if declared_feature_count is not None and feature_audit.get("feature_count") != declared_feature_count:
                errors.append(f"{field}_feature_count_mismatch")
            declared_rows = dataset_manifest.get("output_rows")
            if declared_rows is not None and feature_audit.get("dataset_rows") != declared_rows:
                errors.append(f"{field}_dataset_row_count_mismatch")

    # Verify the dataset manifest's own canonical fingerprint and source linkage
    # when the newer manifest fields are present. Older manifests remain readable.
    if dataset_manifest:
        declared_dataset_manifest_hash = dataset_manifest.get("manifest_sha256")
        if declared_dataset_manifest_hash is not None:
            unhashed_dataset_manifest = dict(dataset_manifest)
            unhashed_dataset_manifest.pop("manifest_sha256", None)
            try:
                actual_dataset_manifest_hash = hashlib.sha256(canonical_json(unhashed_dataset_manifest)).hexdigest()
                if declared_dataset_manifest_hash != actual_dataset_manifest_hash:
                    errors.append("dataset_manifest_self_hash_mismatch")
            except (TypeError, ValueError):
                errors.append("dataset_manifest_not_canonical_json")
        raw_hash = checked_artifacts.get("raw_ohlc_path", {}).get("sha256")
        declared_raw_hash = dataset_manifest.get("input_snapshot_sha256")
        if declared_raw_hash is not None and declared_raw_hash != raw_hash:
            errors.append("dataset_manifest_input_snapshot_hash_mismatch")
        for dataset_field, pipeline_field, error_name in (
            ("feature_schema_sha256", "feature_schema_sha256", "dataset_manifest_feature_schema_hash_mismatch"),
            ("feature_implementation_sha256", "feature_implementation_sha256", "dataset_manifest_feature_implementation_hash_mismatch"),
        ):
            pipeline_value = manifest.get(pipeline_field)
            if pipeline_value is not None and dataset_manifest.get(dataset_field) is not None and dataset_manifest.get(dataset_field) != pipeline_value:
                errors.append(error_name)

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
            completion = snapshot_json(completion_path)
            checked_artifacts["pipeline_completion_audit_path"] = {
                "path": str(completion_path.resolve()),
                "sha256": snapshot_hash(completion_path),
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
                        actual_hash = snapshot_hash(source_path)
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

    # The audit is only internally consistent if no observed artifact changed
    # after its snapshot was captured, even if a mutation was later restored.
    for observed_path, original_bytes in snapshots.items():
        try:
            if observed_path.read_bytes() != original_bytes:
                errors.append(f"artifact_changed_during_audit:{observed_path.name}")
        except OSError:
            errors.append(f"artifact_unreadable_at_audit_end:{observed_path.name}")

    if not checked_artifacts:
        warnings.append("no_artifacts_were_verified")

    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": snapshot_hash(manifest_path),
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

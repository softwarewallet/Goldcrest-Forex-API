"""Cross-artifact reproducibility audit for completed offline research runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads
from app.research_experiment_fingerprint import canonical_json, verify_experiment_fingerprint


def _read_object(path: Path) -> dict[str, Any]:
    """Read a strict JSON object; malformed or ambiguous inputs fail closed."""
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("json_root_must_be_object")
    return value


def _load_json_object(raw: bytes, *, source: str = "<bytes>") -> dict[str, Any]:
    """Parse strict JSON bytes for compatibility with existing audit callers."""
    value = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        relative_to_manifest = manifest_path.parent / candidate
        if relative_to_manifest.is_file():
            return relative_to_manifest.resolve()
    return candidate.resolve()


def _check_file_hash(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    path_field: str,
    hash_field: str,
    check_name: str,
    checks: dict[str, str],
    errors: list[str],
) -> Path | None:
    path = _resolve_artifact(manifest.get(path_field), manifest_path)
    if path is None or not path.is_file():
        checks[check_name] = "FAIL"
        errors.append(f"artifact_missing:{path_field}")
        return None
    try:
        actual = _sha256(path)
    except OSError:
        checks[check_name] = "FAIL"
        errors.append(f"artifact_unreadable:{path_field}")
        return None
    if manifest.get(hash_field) != actual:
        checks[check_name] = "FAIL"
        errors.append(f"artifact_hash_mismatch:{hash_field}")
    else:
        checks[check_name] = "PASS"
    return path


def audit_experiment_reproducibility(
    fingerprint_path: str | Path,
    pipeline_manifest_path: str | Path,
) -> dict[str, Any]:
    """Check fingerprint self-integrity, pipeline linkage, and referenced artifact bytes."""
    fp_path = Path(fingerprint_path).expanduser().resolve()
    pipe_path = Path(pipeline_manifest_path).expanduser().resolve()
    errors: list[str] = []
    checks: dict[str, str] = {}
    fingerprint: dict[str, Any] = {}
    manifest: dict[str, Any] = {}

    try:
        fingerprint = _read_object(fp_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"fingerprint_unreadable:{type(exc).__name__}")
    try:
        manifest = _read_object(pipe_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"pipeline_manifest_unreadable:{type(exc).__name__}")

    if fingerprint:
        verification = verify_experiment_fingerprint(fingerprint)
        checks["fingerprint_self_hash"] = verification["status"]
        errors.extend(verification["errors"])
    if manifest:
        if manifest.get("actionable") is not False:
            errors.append("pipeline_actionable_must_be_false")
        if manifest.get("execution_connected") is not False:
            errors.append("pipeline_execution_connected_must_be_false")
        claimed_manifest_hash = manifest.get("pipeline_manifest_sha256")
        unhashed_manifest = dict(manifest)
        unhashed_manifest.pop("pipeline_manifest_sha256", None)
        try:
            actual_manifest_hash = hashlib.sha256(
                json.dumps(
                    unhashed_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
        except ValueError:
            actual_manifest_hash = None
            errors.append("pipeline_manifest_not_canonicalizable")
        checks["pipeline_manifest_self_hash"] = (
            "PASS" if claimed_manifest_hash == actual_manifest_hash else "FAIL"
        )
        if claimed_manifest_hash != actual_manifest_hash:
            errors.append("pipeline_manifest_sha256_mismatch")

    if fingerprint and manifest:
        if manifest.get("experiment_fingerprint_sha256") != fingerprint.get("experiment_fingerprint_sha256"):
            checks["pipeline_fingerprint_link"] = "FAIL"
            errors.append("pipeline_fingerprint_hash_mismatch")
        else:
            checks["pipeline_fingerprint_link"] = "PASS"

        actual_fingerprint_file_hash = _sha256(fp_path)
        if manifest.get("experiment_fingerprint_file_sha256") != actual_fingerprint_file_hash:
            checks["fingerprint_file_hash_link"] = "FAIL"
            errors.append("pipeline_fingerprint_file_hash_mismatch")
        else:
            checks["fingerprint_file_hash_link"] = "PASS"

        recorded_fingerprint_path = _resolve_artifact(manifest.get("experiment_fingerprint_path"), pipe_path)
        if recorded_fingerprint_path != fp_path:
            checks["fingerprint_path_link"] = "FAIL"
            errors.append("pipeline_fingerprint_path_mismatch")
        else:
            checks["fingerprint_path_link"] = "PASS"

        for key in ("raw_ohlc_sha256", "dataset_sha256"):
            if fingerprint.get(key) != manifest.get(key):
                checks[key] = "FAIL"
                errors.append(f"{key}_mismatch_between_fingerprint_and_pipeline")
            else:
                checks[key] = "PASS"

        raw_path = _check_file_hash(
            manifest, pipe_path, path_field="raw_ohlc_path", hash_field="raw_ohlc_sha256",
            check_name="raw_ohlc_file_hash", checks=checks, errors=errors,
        )
        dataset_path = _check_file_hash(
            manifest, pipe_path, path_field="dataset_path", hash_field="dataset_sha256",
            check_name="dataset_file_hash", checks=checks, errors=errors,
        )
        dataset_manifest_path = _check_file_hash(
            manifest, pipe_path, path_field="dataset_manifest_path", hash_field="dataset_manifest_sha256",
            check_name="dataset_manifest_file_hash", checks=checks, errors=errors,
        )

        if dataset_manifest_path is not None:
            try:
                dataset_manifest = _read_object(dataset_manifest_path)
                unhashed_dataset_manifest = dict(dataset_manifest)
                declared_self_hash = unhashed_dataset_manifest.pop("manifest_sha256", None)
                actual_content_hash = hashlib.sha256(canonical_json(unhashed_dataset_manifest)).hexdigest()
                manifest_file_hash = _sha256(dataset_manifest_path)
                matches = (
                    declared_self_hash == actual_content_hash
                    and manifest_file_hash == fingerprint.get("dataset_manifest_file_sha256")
                    and actual_content_hash == fingerprint.get("dataset_manifest_content_sha256")
                    and dataset_manifest.get("input_snapshot_sha256") == fingerprint.get("raw_ohlc_sha256")
                    and dataset_manifest.get("output_dataset_sha256") == fingerprint.get("dataset_sha256")
                    and dataset_manifest.get("feature_implementation_sha256") == fingerprint.get("feature_implementation_sha256")
                    and dataset_manifest.get("feature_schema_sha256") == fingerprint.get("feature_schema_sha256")
                )
                checks["dataset_manifest_fingerprint_link"] = "PASS" if matches else "FAIL"
                if not matches:
                    errors.append("dataset_manifest_fingerprint_mismatch")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                checks["dataset_manifest_fingerprint_link"] = "FAIL"
                errors.append("dataset_manifest_unreadable")

        # The feature-source receipt is a separate artifact and must match the exact
        # receipt hash and PASS status recorded in the immutable pipeline manifest.
        feature_audit_path = _check_file_hash(
            manifest, pipe_path, path_field="feature_source_audit_path",
            hash_field="feature_source_audit_sha256", check_name="feature_source_audit_file_hash",
            checks=checks, errors=errors,
        )
        if feature_audit_path is not None:
            try:
                feature_audit = _read_object(feature_audit_path)
                valid = (
                    feature_audit.get("status") == "PASS"
                    and feature_audit.get("errors") == []
                    and manifest.get("feature_source_audit_status") == "PASS"
                    and feature_audit.get("dataset_sha256") == manifest.get("dataset_sha256")
                )
                checks["feature_source_audit_status"] = "PASS" if valid else "FAIL"
                if not valid:
                    errors.append("feature_source_audit_not_passing_or_linked")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                checks["feature_source_audit_status"] = "FAIL"
                errors.append("feature_source_audit_unreadable")

        artifact_pairs = (
            ("raw_ohlc_audit_path", "raw_ohlc_audit_sha256", "raw_ohlc_audit_file_hash"),
            ("experiment_matrix_report", "experiment_matrix_report_sha256", "experiment_matrix_file_hash"),
            ("provenance_audit_path", "provenance_audit_sha256", "provenance_audit_file_hash"),
            ("temporal_calibration_audit_path", "temporal_calibration_audit_sha256", "temporal_calibration_file_hash"),
            ("model_evaluation_gate_path", "model_evaluation_gate_sha256", "model_evaluation_gate_file_hash"),
        )
        for path_field, hash_field, check_name in artifact_pairs:
            _check_file_hash(
                manifest, pipe_path, path_field=path_field, hash_field=hash_field,
                check_name=check_name, checks=checks, errors=errors,
            )

        configuration = fingerprint.get("configuration")
        if not isinstance(configuration, dict):
            errors.append("fingerprint_configuration_missing")
        else:
            top_level_fields = (
                "symbol", "timeframe", "source", "source_version", "retrieved_at_utc",
                "label_horizon_bars", "class_1_means", "max_label_elapsed_hours",
            )
            for field in top_level_fields:
                if configuration.get(field) != manifest.get(field):
                    errors.append(f"configuration_mismatch:{field}")
            if dataset_manifest_path is not None:
                try:
                    dataset_manifest = _read_object(dataset_manifest_path)
                    if configuration.get("terms_reference") != dataset_manifest.get("terms_reference"):
                        errors.append("configuration_mismatch:terms_reference")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    pass

            research_config = manifest.get("research_config")
            if not isinstance(research_config, dict):
                errors.append("pipeline_research_config_missing")
            else:
                for field in (
                    "initial_train_bars", "validation_bars", "test_bars", "step_bars",
                    "trials_per_fold", "random_seed", "bootstrap_resamples", "max_experiments",
                    "include_ablation", "strict_provenance_audit", "strict_validation",
                    "strict_temporal_calibration_audit",
                ):
                    if configuration.get(field) != research_config.get(field):
                        errors.append(f"configuration_mismatch:{field}")
            if configuration.get("strict_model_evaluation_gate") != manifest.get("strict_model_evaluation_gate"):
                errors.append("configuration_mismatch:strict_model_evaluation_gate")

        implementation_hashes = fingerprint.get("implementation_file_sha256")
        if isinstance(implementation_hashes, dict):
            repository_root = Path(__file__).resolve().parents[1]
            for relative_name, expected_hash in implementation_hashes.items():
                relative_path = Path(relative_name)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    errors.append(f"implementation_path_invalid:{relative_name}")
                    continue
                implementation_path = repository_root / relative_path
                try:
                    actual_hash = _sha256(implementation_path)
                except OSError:
                    errors.append(f"implementation_file_missing:{relative_name}")
                    continue
                if actual_hash != expected_hash:
                    errors.append(f"implementation_file_changed_since_fingerprint:{relative_name}")
        else:
            errors.append("implementation_file_sha256_missing")

    return {
        "schema_version": 1,
        "audit_name": "research_experiment_reproducibility_audit",
        "scope": "OFFLINE_RESEARCH_REPRODUCIBILITY_AUDIT",
        "status": "FAIL" if errors else "PASS",
        "errors": sorted(set(errors)),
        "checks": checks,
        "fingerprint_path": str(fp_path),
        "fingerprint_file_sha256": _sha256(fp_path) if fp_path.is_file() else None,
        "pipeline_manifest_path": str(pipe_path),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "interpretation": (
            "This audit checks linked identities, hashes, configuration, and current source files. "
            "It does not prove source correctness, predictive validity, or profitability."
        ),
    }

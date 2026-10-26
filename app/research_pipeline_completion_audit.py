"""Final integrity gate for completed offline research-pipeline runs.

This gate consumes persisted reports only. It does not rerun experiments, train models,
select a winner, connect to a broker, or authorize deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

_FALSE_FLAGS = (
    "actionable",
    "execution_connected",
    "automatic_winner_selection",
    "test_metrics_used_for_selection",
)
_PASS = {"PASS", "PASS_WITH_WARNINGS"}


def _read_object_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and fingerprint one byte snapshot so the digest matches the audited content."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path.name}")
    return payload, digest


def _read_object(path: Path) -> dict[str, Any]:
    return _read_object_with_hash(path)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _resolve(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        relative = manifest_path.parent / candidate
        if relative.is_file():
            return relative.resolve()
    return candidate.resolve()


def audit_pipeline_completion(
    manifest_path: str | Path,
    validation_report_path: str | Path,
    reproducibility_audit_path: str | Path,
) -> dict[str, Any]:
    """Verify the final persisted run reports and their links to the immutable manifest."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation_file = Path(validation_report_path).expanduser().resolve()
    reproducibility_file = Path(reproducibility_audit_path).expanduser().resolve()
    errors: list[dict[str, str]] = []
    warnings: list[str] = []
    checks: dict[str, str] = {}
    manifest: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    reproducibility: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    evaluation_source_hash: str | None = None

    for label, path, target in (
        ("pipeline_manifest", manifest_file, "manifest"),
        ("validation_report", validation_file, "validation"),
        ("reproducibility_audit", reproducibility_file, "reproducibility"),
    ):
        try:
            value, source_hash = _read_object_with_hash(path)
            source_hashes[label] = source_hash
            if target == "manifest":
                manifest = value
            elif target == "validation":
                validation = value
            else:
                reproducibility = value
            checks[f"{label}_readable"] = "PASS"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            checks[f"{label}_readable"] = "FAIL"
            errors.append({"code": "required_report_unreadable", "report": label})

    if not manifest:
        return _report(
            manifest, errors, warnings, checks, validation, reproducibility, evaluation,
            validation_report_sha256=source_hashes.get("validation_report"),
            reproducibility_audit_sha256=source_hashes.get("reproducibility_audit"),
        )

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append({"code": "manifest_run_id_invalid", "report": "pipeline_manifest"})

    expected_manifest_hash = manifest.get("pipeline_manifest_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("pipeline_manifest_sha256", None)
    try:
        actual_manifest_hash = hashlib.sha256(
            json.dumps(unhashed_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        actual_manifest_hash = None
    checks["pipeline_manifest_self_hash"] = (
        "PASS" if expected_manifest_hash == actual_manifest_hash else "FAIL"
    )
    if checks["pipeline_manifest_self_hash"] == "FAIL":
        errors.append({"code": "pipeline_manifest_sha256_mismatch", "report": "pipeline_manifest"})

    for flag in _FALSE_FLAGS:
        if manifest.get(flag) is not False:
            errors.append({"code": "safety_flag_not_explicitly_false", "report": "pipeline_manifest", "field": flag})

    if validation:
        if validation.get("scope") != "GOLDCREST_COMPLETED_RUN_VALIDATION":
            errors.append({"code": "validation_report_scope_mismatch", "report": "validation_report"})
        if validation.get("run_id") != run_id:
            errors.append({"code": "validation_report_run_id_mismatch", "report": "validation_report"})
        if validation.get("manifest_path") and Path(str(validation["manifest_path"])).expanduser().resolve() != manifest_file:
            errors.append({"code": "validation_report_manifest_path_mismatch", "report": "validation_report"})
        validation_status = validation.get("status")
        if not isinstance(validation_status, str) or validation_status not in {"PASS", "PASS_WITH_WARNINGS", "FAIL"}:
            errors.append({"code": "validation_report_status_schema_invalid", "report": "validation_report"})
        validation_errors = validation.get("errors")
        if not isinstance(validation_errors, list):
            errors.append({"code": "validation_report_errors_schema_invalid", "report": "validation_report"})
        elif validation_status in {"FAIL", "PASS_WITH_WARNINGS"} or validation_errors:
            warnings.append("validation_report_contains_quality_or_integrity_findings")
        validation_warnings = validation.get("warnings")
        if (
            not isinstance(validation_warnings, list)
            or any(not isinstance(item, str) or not item.strip() for item in validation_warnings)
        ):
            errors.append({"code": "validation_report_warnings_schema_invalid", "report": "validation_report"})
        elif validation_status == "PASS" and validation_warnings:
            errors.append({"code": "validation_report_status_warnings_mismatch", "report": "validation_report"})
        elif validation_status == "PASS_WITH_WARNINGS" and not validation_warnings:
            errors.append({"code": "validation_report_status_warnings_mismatch", "report": "validation_report"})
        if validation.get("actionable") is not False or validation.get("execution_connected") is not False:
            errors.append({"code": "validation_report_safety_flags_invalid", "report": "validation_report"})
        if validation_status == "PASS_WITH_WARNINGS" or validation_warnings:
            warnings.append("validation_report_has_warnings")
        checks["validation_report_status"] = (
            "PASS" if isinstance(validation_status, str) and validation_status in _PASS and validation_errors == [] else "WARN"
        )
    else:
        checks["validation_report_status"] = "FAIL"

    if reproducibility:
        if reproducibility.get("audit_name") != "research_experiment_reproducibility_audit":
            errors.append({"code": "reproducibility_audit_name_mismatch", "report": "reproducibility_audit"})
        if reproducibility.get("status") != "PASS" or reproducibility.get("errors") != []:
            errors.append({"code": "reproducibility_audit_not_passing", "report": "reproducibility_audit"})
        if reproducibility.get("pipeline_manifest_path") is None or Path(
            str(reproducibility.get("pipeline_manifest_path"))
        ).expanduser().resolve() != manifest_file:
            errors.append({"code": "reproducibility_manifest_path_mismatch", "report": "reproducibility_audit"})
        if reproducibility.get("actionable") is not False or reproducibility.get("execution_connected") is not False:
            errors.append({"code": "reproducibility_safety_flags_invalid", "report": "reproducibility_audit"})
        recorded_fp = _resolve(manifest.get("experiment_fingerprint_path"), manifest_file)
        if recorded_fp is None or reproducibility.get("fingerprint_path") is None or Path(
            str(reproducibility.get("fingerprint_path"))
        ).expanduser().resolve() != recorded_fp:
            errors.append({"code": "reproducibility_fingerprint_path_mismatch", "report": "reproducibility_audit"})
        expected_fp_hash = reproducibility.get("fingerprint_file_sha256")
        if recorded_fp is None or not recorded_fp.is_file() or expected_fp_hash != _sha256(recorded_fp):
            errors.append({"code": "reproducibility_fingerprint_hash_mismatch", "report": "reproducibility_audit"})
        checks["reproducibility_audit_status"] = "PASS" if reproducibility.get("status") == "PASS" and not reproducibility.get("errors") else "FAIL"
    else:
        checks["reproducibility_audit_status"] = "FAIL"

    evaluation_path = _resolve(manifest.get("model_evaluation_gate_path"), manifest_file)
    expected_evaluation_hash = manifest.get("model_evaluation_gate_sha256")
    try:
        if evaluation_path is None or not evaluation_path.is_file():
            raise OSError("model_evaluation_gate_missing")
        evaluation, evaluation_source_hash = _read_object_with_hash(evaluation_path)
        if evaluation_source_hash != expected_evaluation_hash:
            raise ValueError("model_evaluation_gate_sha256_mismatch")
        checks["model_evaluation_gate_file_hash"] = "PASS"
        if evaluation.get("run_id") != run_id:
            errors.append({"code": "model_evaluation_gate_run_id_mismatch", "report": "model_evaluation_gate"})
        expected_matrix_sha256 = manifest.get("experiment_matrix_report_sha256")
        if expected_matrix_sha256 is not None:
            if not _is_sha256_hex(expected_matrix_sha256):
                errors.append({"code": "manifest_experiment_matrix_sha256_invalid", "report": "pipeline_manifest"})
            elif evaluation.get("source_matrix_sha256") != expected_matrix_sha256:
                errors.append({"code": "model_evaluation_gate_source_matrix_sha256_mismatch", "report": "model_evaluation_gate"})
        evaluation_errors = evaluation.get("errors")
        if not isinstance(evaluation_errors, list):
            errors.append({"code": "model_evaluation_gate_errors_schema_invalid", "report": "model_evaluation_gate"})
        else:
            if not isinstance(evaluation.get("status"), str) or evaluation.get("status") not in _PASS:
                warnings.append("model_evaluation_gate_did_not_pass")
            if evaluation_errors:
                warnings.append("model_evaluation_gate_reports_errors")
        evaluation_status = evaluation.get("status")
        if not isinstance(evaluation_status, str) or evaluation_status not in {"PASS", "PASS_WITH_WARNINGS", "FAIL"}:
            errors.append({"code": "model_evaluation_gate_status_schema_invalid", "report": "model_evaluation_gate"})
        evaluation_warnings = evaluation.get("warnings")
        if (
            not isinstance(evaluation_warnings, list)
            or any(not isinstance(item, str) or not item.strip() for item in evaluation_warnings)
        ):
            errors.append({"code": "model_evaluation_gate_warnings_schema_invalid", "report": "model_evaluation_gate"})
        elif evaluation_status == "PASS" and evaluation_warnings:
            errors.append({"code": "model_evaluation_gate_status_warnings_mismatch", "report": "model_evaluation_gate"})
        elif evaluation_status == "PASS_WITH_WARNINGS" and not evaluation_warnings:
            errors.append({"code": "model_evaluation_gate_status_warnings_mismatch", "report": "model_evaluation_gate"})
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        checks["model_evaluation_gate_file_hash"] = "FAIL"
        errors.append({"code": "model_evaluation_gate_missing_unreadable_or_tampered", "report": "model_evaluation_gate"})

    for label, path in (
        ("validation_report", validation_file),
        ("reproducibility_audit", reproducibility_file),
    ):
        if not path.is_file():
            checks[f"{label}_file_hash"] = "FAIL"
            errors.append({"code": "required_report_file_missing", "report": label})
        elif label not in source_hashes:
            checks[f"{label}_file_hash"] = "FAIL"
            errors.append({"code": "required_report_hash_unavailable", "report": label})
        else:
            # The digest was computed from the exact JSON bytes parsed above.
            checks[f"{label}_file_hash"] = "PASS"

    return _report(
        manifest, errors, warnings, checks, validation, reproducibility, evaluation,
        validation_report_sha256=source_hashes.get("validation_report"),
        reproducibility_audit_sha256=source_hashes.get("reproducibility_audit"),
        model_evaluation_gate_sha256=evaluation_source_hash,
    )


def _report(
    manifest, errors, warnings, checks, validation, reproducibility, evaluation,
    *, validation_report_sha256=None, reproducibility_audit_sha256=None,
    model_evaluation_gate_sha256=None,
):
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": 2,
        "audit_name": "research_pipeline_completion_audit",
        "scope": "OFFLINE_RESEARCH_PIPELINE_COMPLETION",
        "status": status,
        "readiness": (
            "NOT_VERIFIED" if errors else
            "RUN_INTEGRITY_VERIFIED_WITH_WARNINGS" if warnings else
            "RUN_INTEGRITY_VERIFIED"
        ),
        "run_id": manifest.get("run_id") if isinstance(manifest, dict) else None,
        "pipeline_manifest_sha256": manifest.get("pipeline_manifest_sha256") if isinstance(manifest, dict) else None,
        "validation_status": validation.get("status") if isinstance(validation, dict) else None,
        "reproducibility_status": reproducibility.get("status") if isinstance(reproducibility, dict) else None,
        "model_evaluation_status": evaluation.get("status") if isinstance(evaluation, dict) else None,
        "validation_report_sha256": validation_report_sha256,
        "reproducibility_audit_sha256": reproducibility_audit_sha256,
        "model_evaluation_gate_sha256": model_evaluation_gate_sha256,
        "model_qualified": False,
        "checks": checks,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "This receipt verifies persisted report consistency and selected artifact links; it does not rerun model training or all upstream calculations.",
            "A passing integrity receipt does not establish predictive validity, profitability, statistical significance, or deployment suitability.",
            "A model evaluation gate failure is surfaced as a warning because this receipt certifies run integrity, not model qualification; inspect model_evaluation_status separately.",
            "The receipt is written after the pipeline manifest to avoid a circular hash dependency and is not itself included in that manifest.",
        ],
    }



def _write_report_no_clobber(output_path: Path, report: dict[str, Any]) -> None:
    """Publish one JSON receipt atomically without replacing an existing file."""
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link fails if another process created the destination after preflight.
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-audit completion integrity for an existing offline research run."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--reproducibility-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    inputs = {
        args.manifest.expanduser().resolve(),
        args.validation_report.expanduser().resolve(),
        args.reproducibility_audit.expanduser().resolve(),
    }
    try:
        if output in inputs:
            raise ValueError("output_must_not_alias_input")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        report = audit_pipeline_completion(
            args.manifest, args.validation_report, args.reproducibility_audit
        )
        _write_report_no_clobber(output, report)
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "run_id": report["run_id"],
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "report_path": str(output),
            "model_qualified": False,
            "deployment_authorized": False,
        }, indent=2))
        return 1 if report["status"] == "FAIL" else 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL",
            "error": f"{type(exc).__name__}:{exc}",
            "model_qualified": False,
            "deployment_authorized": False,
        }, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

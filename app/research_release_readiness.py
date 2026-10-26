"""Fail-closed release-readiness qualification for offline research artifacts.

This gate verifies persisted artifact integrity and explicit provider-terms review evidence.
It does not decide legal applicability, validate predictive performance, authorize deployment,
or connect to any execution system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.research_run_audit import audit_research_run_manifest
from app.research_strict_json import strict_json_loads

_FALSE_FLAGS = (
    "actionable",
    "execution_connected",
    "automatic_winner_selection",
    "test_metrics_used_for_selection",
    "automatic_model_selection",
    "deployment_authorized",
)


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _resolve_path(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def audit_research_release_readiness(
    manifest_path: str | Path,
    provider_terms_evidence_path: str | Path,
) -> dict[str, Any]:
    """Audit a research run and determine whether its evidence is release-ready.

    A passing result means the recorded integrity and evidence checklist passed. It is not
    a legal opinion, model-quality certification, profitability claim, or deployment approval.
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    evidence_file = Path(provider_terms_evidence_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, str] = {}

    try:
        manifest, manifest_bytes = _read_object(manifest_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return _result(
            manifest_path=str(manifest_file),
            evidence_path=str(evidence_file),
            errors=[f"pipeline_manifest_unreadable:{type(exc).__name__}"],
            warnings=[],
            checks={"pipeline_manifest": "FAIL"},
        )

    manifest_hash = _sha256(manifest_bytes)
    run_audit = audit_research_run_manifest(manifest_file)
    checks["cross_artifact_integrity"] = run_audit.get("status", "FAIL")
    if run_audit.get("status") == "FAIL":
        errors.append("cross_artifact_integrity_failed")
    elif run_audit.get("status") == "PASS_WITH_WARNINGS":
        warnings.extend(f"integrity:{item}" for item in run_audit.get("warnings", []))

    for flag in _FALSE_FLAGS:
        if manifest.get(flag) is not False:
            errors.append(f"pipeline_{flag}_must_be_false")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("pipeline_run_id_missing")

    # A consistent URI/hash proves what was recorded, not that the terms are applicable.
    reference = manifest.get("terms_reference")
    declared_terms_hash = manifest.get("terms_reference_sha256")
    if not isinstance(reference, str) or not reference.strip():
        errors.append("provider_terms_reference_missing")
    else:
        normalized_reference = reference.strip()
        if re.search(
            r"<[^>]+>|\\b(?:TODO|TBD|PLACEHOLDER|REPLACE THIS|EXACT PROVIDER TERMS REFERENCE)\\b",
            normalized_reference,
            flags=re.IGNORECASE,
        ):
            errors.append("provider_terms_reference_is_placeholder")
        actual_terms_hash = hashlib.sha256(normalized_reference.encode("utf-8")).hexdigest()
        if declared_terms_hash != actual_terms_hash:
            errors.append("provider_terms_reference_hash_mismatch")
        if not re.match(r"^https?://[^\\s]+$", normalized_reference, flags=re.IGNORECASE):
            errors.append("provider_terms_reference_must_be_http_or_https_uri")

    try:
        evidence, evidence_bytes = _read_object(evidence_file)
        evidence_hash = _sha256(evidence_bytes)
        checks["provider_terms_evidence_json"] = "PASS"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        evidence = {}
        evidence_hash = None
        checks["provider_terms_evidence_json"] = "FAIL"
        errors.append(f"provider_terms_evidence_unreadable:{type(exc).__name__}")

    if evidence:
        if evidence.get("schema_version") != 1:
            errors.append("provider_terms_evidence_schema_unsupported")
        if evidence.get("evidence_type") != "provider_terms_review":
            errors.append("provider_terms_evidence_type_invalid")
        if evidence.get("review_status") != "VERIFIED":
            errors.append("provider_terms_review_not_verified")
        if evidence.get("applicability") != "CONFIRMED":
            errors.append("provider_terms_applicability_not_confirmed")
        evidence_reference = evidence.get("terms_reference")
        if not isinstance(evidence_reference, str) or not isinstance(reference, str) or evidence_reference.strip() != reference.strip():
            errors.append("provider_terms_evidence_reference_mismatch")
        if evidence.get("terms_reference_sha256") != declared_terms_hash:
            errors.append("provider_terms_evidence_reference_hash_mismatch")
        provider = evidence.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            errors.append("provider_terms_evidence_provider_missing")
        reviewer = evidence.get("reviewed_by")
        if not isinstance(reviewer, str) or not reviewer.strip():
            errors.append("provider_terms_evidence_reviewer_missing")
        if not _valid_utc_timestamp(evidence.get("reviewed_at_utc")):
            errors.append("provider_terms_evidence_review_timestamp_invalid")

        source_path = _resolve_path(evidence.get("source_evidence_path"), evidence_file.parent)
        if source_path is None or not source_path.is_file():
            errors.append("provider_terms_source_evidence_missing")
            checks["provider_terms_source_evidence"] = "FAIL"
        else:
            try:
                source_hash = _sha256(source_path.read_bytes())
                checks["provider_terms_source_evidence"] = "PASS"
                declared_source_hash = evidence.get("source_evidence_sha256")
                if not isinstance(declared_source_hash, str) or declared_source_hash.lower() != source_hash:
                    errors.append("provider_terms_source_evidence_hash_mismatch")
            except OSError:
                checks["provider_terms_source_evidence"] = "FAIL"
                errors.append("provider_terms_source_evidence_unreadable")
        if evidence.get("source_evidence_sha256") is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", str(evidence.get("source_evidence_sha256"))
        ):
            errors.append("provider_terms_source_evidence_hash_invalid")

    # Release qualification is stricter than run integrity: warnings, including unresolved
    # raw-data coverage gaps, keep the run out of a release-ready state.
    if run_audit.get("status") == "PASS_WITH_WARNINGS":
        errors.append("release_blocked_by_cross_artifact_warnings")

    required_receipts = (
        ("raw_ohlc_audit_path", "raw_ohlc_audit_status"),
        ("feature_source_audit_path", "feature_source_audit_status"),
        ("point_in_time_feature_invariance_audit_path", "point_in_time_feature_invariance_audit_status"),
        ("label_maturity_audit_path", "label_maturity_audit_status"),
        ("provenance_audit_path", "provenance_audit_status"),
        ("temporal_calibration_audit_path", "temporal_calibration_audit_status"),
        ("model_evaluation_gate_path", "model_evaluation_gate_status"),
    )
    for path_field, status_field in required_receipts:
        if manifest.get(status_field) != "PASS":
            errors.append(f"required_{status_field}_not_clean_pass")
        receipt_path = _resolve_path(manifest.get(path_field), manifest_file.parent)
        if receipt_path is None or not receipt_path.is_file():
            errors.append(f"required_receipt_missing:{path_field}")
            checks[path_field] = "FAIL"
            continue
        try:
            receipt, receipt_bytes = _read_object(receipt_path)
            checks[path_field] = "PASS"
            if receipt.get("status") != "PASS":
                errors.append(f"required_receipt_status_not_clean_pass:{path_field}")
            if receipt.get("errors") != []:
                errors.append(f"required_receipt_contains_errors:{path_field}")
            for safety_flag in ("actionable", "execution_connected", "automatic_model_selection",
                                "automatic_winner_selection", "test_metrics_used_for_selection"):
                if safety_flag in receipt and receipt.get(safety_flag) is not False:
                    errors.append(f"required_receipt_safety_flag_invalid:{path_field}:{safety_flag}")
            expected_hash_field = {
                "raw_ohlc_audit_path": "raw_ohlc_audit_sha256",
                "feature_source_audit_path": "feature_source_audit_sha256",
                "point_in_time_feature_invariance_audit_path": "point_in_time_feature_invariance_audit_sha256",
                "label_maturity_audit_path": "label_maturity_audit_sha256",
                "provenance_audit_path": "provenance_audit_sha256",
                "temporal_calibration_audit_path": "temporal_calibration_audit_sha256",
                "model_evaluation_gate_path": "model_evaluation_gate_sha256",
            }[path_field]
            if _sha256(receipt_bytes) != manifest.get(expected_hash_field):
                errors.append(f"required_receipt_file_hash_mismatch:{path_field}")
            if receipt.get("run_id") is not None and receipt.get("run_id") != run_id:
                errors.append(f"required_receipt_run_id_mismatch:{path_field}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            checks[path_field] = "FAIL"
            errors.append(f"required_receipt_unreadable:{path_field}:{type(exc).__name__}")

    reproducibility_path = _resolve_path(manifest.get("reproducibility_audit_path"), manifest_file.parent)
    if reproducibility_path is None or not reproducibility_path.is_file():
        errors.append("required_reproducibility_audit_missing")
        checks["reproducibility_audit"] = "FAIL"
    else:
        try:
            reproducibility, reproducibility_bytes = _read_object(reproducibility_path)
            checks["reproducibility_audit"] = "PASS"
            if reproducibility.get("status") != "PASS" or reproducibility.get("errors") != []:
                errors.append("required_reproducibility_audit_not_clean_pass")
            if reproducibility.get("actionable") is not False or reproducibility.get("execution_connected") is not False:
                errors.append("required_reproducibility_audit_safety_flags_invalid")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            checks["reproducibility_audit"] = "FAIL"
            errors.append(f"required_reproducibility_audit_unreadable:{type(exc).__name__}")

    completion_path = manifest_file.parent / f"{run_id}_pipeline_completion_audit.json" if isinstance(run_id, str) else None
    if completion_path is None or not completion_path.is_file():
        errors.append("pipeline_completion_receipt_missing")
        checks["pipeline_completion_receipt"] = "FAIL"
    else:
        try:
            completion, completion_bytes = _read_object(completion_path)
            checks["pipeline_completion_receipt"] = "PASS"
            if completion.get("run_id") != run_id:
                errors.append("pipeline_completion_receipt_run_id_mismatch")
            if completion.get("pipeline_manifest_sha256") != manifest.get("pipeline_manifest_sha256"):
                errors.append("pipeline_completion_receipt_manifest_hash_mismatch")
            if completion.get("status") != "PASS" or completion.get("errors") != [] or completion.get("warnings") != []:
                errors.append("pipeline_completion_receipt_not_clean_pass")
            if completion.get("schema_version") != 2:
                errors.append("pipeline_completion_receipt_schema_not_source_bound")
            source_specs = (
                ("validation_report_sha256", manifest_file.parent / f"{run_id}_validation_report.json"),
                ("reproducibility_audit_sha256", reproducibility_path),
                ("model_evaluation_gate_sha256", _resolve_path(manifest.get("model_evaluation_gate_path"), manifest_file.parent)),
            )
            for hash_field, source_path in source_specs:
                if source_path is None or not source_path.is_file():
                    errors.append(f"pipeline_completion_source_missing:{hash_field}")
                else:
                    try:
                        if completion.get(hash_field) != _sha256(source_path.read_bytes()):
                            errors.append(f"pipeline_completion_source_hash_mismatch:{hash_field}")
                    except OSError:
                        errors.append(f"pipeline_completion_source_unreadable:{hash_field}")
            for flag in ("actionable", "execution_connected", "model_qualified", "deployment_authorized"):
                if completion.get(flag) is not False:
                    errors.append(f"pipeline_completion_{flag}_must_be_false")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            checks["pipeline_completion_receipt"] = "FAIL"
            errors.append(f"pipeline_completion_receipt_unreadable:{type(exc).__name__}")

    if not errors and not warnings:
        status = "PASS"
        readiness = "RELEASE_READY_FOR_HUMAN_REVIEW"
    else:
        status = "FAIL"
        readiness = "NOT_RELEASE_READY"
    return {
        "schema_version": 1,
        "audit_name": "research_release_readiness_audit",
        "status": status,
        "readiness": readiness,
        "run_id": run_id,
        "pipeline_manifest_path": str(manifest_file),
        "pipeline_manifest_file_sha256": manifest_hash,
        "pipeline_manifest_declared_sha256": manifest.get("pipeline_manifest_sha256"),
        "provider_terms_evidence_path": str(evidence_file),
        "provider_terms_evidence_sha256": evidence_hash,
        "cross_artifact_integrity_status": run_audit.get("status"),
        "cross_artifact_integrity_errors": run_audit.get("errors", []),
        "cross_artifact_integrity_warnings": run_audit.get("warnings", []),
        "checks": checks,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "dataset_fingerprints": (
            {manifest["dataset_sha256"]: [run_id]}
            if isinstance(manifest.get("dataset_sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", manifest["dataset_sha256"])
            else {}
        ),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
        "model_qualified": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "Provider-terms applicability is represented by an explicit human-review evidence record; the tool does not independently decide legal applicability.",
            "A release-ready integrity result does not establish predictive validity, profitability, statistical significance, absence of all leakage, or suitability for live trading.",
            "Unresolved data-quality warnings, missing evidence, non-clean audit statuses, and missing completion receipts fail closed.",
        ],
    }


def _result(*, manifest_path: str, evidence_path: str, errors: list[str],
            warnings: list[str], checks: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_name": "research_release_readiness_audit",
        "status": "FAIL",
        "readiness": "NOT_RELEASE_READY",
        "pipeline_manifest_path": manifest_path,
        "provider_terms_evidence_path": evidence_path,
        "checks": checks,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "model_qualified": False,
        "deployment_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed release-readiness audit for a completed offline research run."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provider-terms-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="Print the report to stdout as JSON.")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    inputs = {args.manifest.expanduser().resolve(), args.provider_terms_evidence.expanduser().resolve()}
    if output in inputs:
        print("output_must_not_alias_input", file=sys.stderr)
        return 3
    if output.exists():
        print(f"output_already_exists:{output}", file=sys.stderr)
        return 3
    report = audit_research_release_readiness(args.manifest, args.provider_terms_evidence)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create avoids overwriting prior qualification evidence.
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except OSError as exc:
        print(f"report_write_failed:{type(exc).__name__}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Research release readiness: {report['status']}")
        print(f"Readiness: {report['readiness']}")
        print(f"Run ID: {report.get('run_id', 'unknown')}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

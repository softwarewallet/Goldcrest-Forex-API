"""End-to-end readiness gate for offline Forex research audit artifacts.

This audit reconciles independently generated JSON reports. It verifies required audit
coverage, report provenance, dataset fingerprint consistency where available, declared
statuses, and safety boundaries. It does not execute upstream audits, inspect arbitrary
pipeline code, train/select models, or authorize trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_AUDITS = (
    "feature_temporal_integrity",
    "feature_backend_parity",
    "pipeline_leakage_boundary",
    "research_artifact_bundle_certification",
)
KNOWN_STATUSES = {"PASS", "PASS_WITH_WARNINGS", "NOT_COMPARABLE", "FAIL"}
REQUIRED_FALSE_FLAGS = (
    "actionable",
    "inferential",
    "execution_connected",
    "model_training_performed",
)
OPTIONAL_FALSE_FLAGS = (
    "automatic_model_selection",
    "automatic_repair_performed",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def audit_research_readiness(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    base_dir: Path,
) -> dict[str, Any]:
    """Reconcile report artifacts into a conservative, non-actionable readiness result."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _valid_sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifacts_must_be_nonempty_list")
    required = manifest.get("required_audits", list(REQUIRED_AUDITS))
    if not isinstance(required, list) or not required or any(
        not isinstance(name, str) or not name for name in required
    ):
        raise ValueError("required_audits_invalid")
    if len(set(required)) != len(required):
        raise ValueError("required_audits_must_be_unique")

    root = base_dir.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    seen_paths: set[str] = set()
    dataset_hashes: dict[str, list[str]] = {}

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"code": "artifact_entry_invalid", "index": index})
            continue

        audit_name = entry.get("audit_name")
        relative_path = entry.get("path")
        entry_errors: list[str] = []
        if not isinstance(audit_name, str) or not audit_name:
            audit_name = f"unknown_{index}"
            entry_errors.append("audit_name_invalid")
        if not isinstance(relative_path, str) or not relative_path:
            relative_path = ""
            entry_errors.append("artifact_path_invalid")

        if audit_name in seen_types:
            entry_errors.append("duplicate_audit_name")
        seen_types.add(audit_name)
        if relative_path in seen_paths:
            entry_errors.append("duplicate_artifact_path")
        if relative_path:
            seen_paths.add(relative_path)

        payload: dict[str, Any] | None = None
        digest: str | None = None
        resolved: Path | None = None
        if relative_path:
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                entry_errors.append("artifact_path_must_be_relative_and_confined")
            else:
                resolved = (root / candidate).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    entry_errors.append("artifact_path_escapes_manifest_directory")
                    resolved = None

        if resolved is not None and not any(
            code in entry_errors for code in (
                "artifact_path_must_be_relative_and_confined",
                "artifact_path_escapes_manifest_directory",
            )
        ):
            try:
                raw = resolved.read_bytes()
                digest = _sha256(raw)
                expected_digest = entry.get("expected_report_sha256")
                if expected_digest is not None:
                    if not _valid_sha256(expected_digest):
                        entry_errors.append("expected_report_sha256_invalid")
                    elif digest != expected_digest.lower():
                        entry_errors.append("report_sha256_mismatch")
                decoded = json.loads(raw.decode("utf-8"))
                if not isinstance(decoded, dict):
                    entry_errors.append("report_root_not_object")
                else:
                    payload = decoded
            except OSError:
                entry_errors.append("artifact_unreadable")
            except (UnicodeDecodeError, json.JSONDecodeError):
                entry_errors.append("artifact_json_invalid")

        status = None
        if payload is not None:
            if payload.get("audit_name") != audit_name:
                entry_errors.append("audit_name_mismatch")
            status = payload.get("status")
            if status not in KNOWN_STATUSES:
                entry_errors.append("status_unknown_or_missing")
            elif status == "FAIL":
                entry_errors.append("upstream_audit_failed")
            elif status == "NOT_COMPARABLE":
                entry_errors.append("upstream_audit_not_comparable")
            elif status == "PASS_WITH_WARNINGS":
                warnings.append(f"upstream_audit_passed_with_warnings:{audit_name}")

            upstream_errors = payload.get("errors")
            if not isinstance(upstream_errors, list):
                entry_errors.append("upstream_errors_field_missing_or_invalid")
            elif upstream_errors:
                entry_errors.append("upstream_report_contains_errors")

            for flag in REQUIRED_FALSE_FLAGS:
                if payload.get(flag) is not False:
                    entry_errors.append(f"required_safety_flag_not_explicitly_false:{flag}")
            for flag in OPTIONAL_FALSE_FLAGS:
                if flag in payload and payload[flag] is not False:
                    entry_errors.append(f"safety_flag_not_false:{flag}")

            dataset_hash = payload.get("dataset_sha256")
            if dataset_hash is None:
                warnings.append(f"dataset_fingerprint_not_exposed:{audit_name}")
            elif not _valid_sha256(dataset_hash):
                entry_errors.append("dataset_sha256_invalid")
            else:
                dataset_hashes.setdefault(dataset_hash.lower(), []).append(audit_name)

            if audit_name == "feature_temporal_integrity":
                if payload.get("prefix_invariance", {}).get("all_passed") is not True:
                    entry_errors.append("prefix_invariance_not_confirmed")
                if payload.get("future_perturbation_invariance", {}).get("all_passed") is not True:
                    entry_errors.append("future_perturbation_invariance_not_confirmed")
                if payload.get("chronological_partition_isolation", {}).get(
                    "purge_gaps_sufficient_for_horizon"
                ) is not True:
                    entry_errors.append("temporal_split_purge_not_confirmed")
            elif audit_name == "feature_backend_parity":
                if payload.get("comparison_performed") is not True:
                    entry_errors.append("backend_comparison_not_performed")
                if payload.get("all_features_passed") is not True:
                    entry_errors.append("all_feature_backend_parity_not_confirmed")
            elif audit_name == "pipeline_leakage_boundary":
                if not isinstance(payload.get("split_results"), dict) or "train" not in payload["split_results"]:
                    entry_errors.append("pipeline_split_evidence_missing")
                if not payload.get("preprocessing_results"):
                    warnings.append("pipeline_preprocessing_evidence_missing")
                if not payload.get("label_results"):
                    warnings.append("pipeline_label_evidence_missing")
            elif audit_name == "research_artifact_bundle_certification":
                if payload.get("missing_required_audits") != []:
                    entry_errors.append("bundle_required_audit_coverage_incomplete")
                if not isinstance(payload.get("artifacts_checked"), int) or payload["artifacts_checked"] < 1:
                    entry_errors.append("bundle_artifact_count_invalid")

        for code in entry_errors:
            errors.append({"code": code, "audit_name": audit_name, "path": relative_path})
        results.append({
            "audit_name": audit_name,
            "path": relative_path,
            "report_sha256": digest,
            "upstream_status": status,
            "passed": not entry_errors,
            "errors": entry_errors,
        })

    missing = sorted(set(required) - seen_types)
    for audit_name in missing:
        errors.append({"code": "required_audit_missing", "audit_name": audit_name})

    if len(dataset_hashes) > 1:
        errors.append({
            "code": "dataset_fingerprint_mismatch_across_reports",
            "dataset_hashes": {
                digest: names for digest, names in sorted(dataset_hashes.items())
            },
        })
    if not dataset_hashes:
        warnings.append("no_dataset_fingerprint_available_for_cross_report_reconciliation")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_readiness_certification",
        "manifest_sha256": manifest_sha256.lower(),
        "required_audits": required,
        "missing_required_audits": missing,
        "artifacts_checked": len(results),
        "artifact_results": results,
        "dataset_fingerprints": {
            digest: names for digest, names in sorted(dataset_hashes.items())
        },
        "errors": errors,
        "warnings": warnings,
        "status": status,
        "readiness": "NOT_READY" if status == "FAIL" else (
            "READY_WITH_WARNINGS" if status == "PASS_WITH_WARNINGS" else "READY"
        ),
        "actionable": False,
        "automatic_model_selection": False,
        "inferential": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "Only supplied report artifacts and their declared evidence are checked.",
            "This result is not proof of profitability, predictive validity, or absence of all leakage.",
            "A passing result does not authorize model deployment or trading execution.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON readiness manifest")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        report = audit_research_readiness(
            manifest,
            manifest_sha256=_sha256(raw),
            base_dir=args.input.resolve().parent,
        )
        output = args.output or args.input.with_name(args.input.stem + "_readiness_certification.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "report_path": str(output.resolve()),
            "artifacts_checked": report["artifacts_checked"],
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

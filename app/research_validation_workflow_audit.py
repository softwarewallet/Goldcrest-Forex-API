"""Consolidated, fail-closed validation of offline Forex research audit reports.

Consumes report artifacts produced by independent audits. It does not run those audits,
read raw market data, train/select models, or authorize deployment or trading.
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
    "walk_forward_robustness",
    "research_cohort_comparability_and_distribution_shift",
    "backtest_result_integrity",
)
KNOWN_STATUSES = {"PASS", "PASS_WITH_WARNINGS", "NOT_COMPARABLE", "FAIL"}
FALSE_FLAGS = (
    "actionable", "inferential", "execution_connected",
    "model_training_performed", "automatic_model_selection",
    "automatic_repair_performed",
)
OPTIONAL_FALSE_FLAGS = ("statistical_significance_claimed",)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in value
    )


def audit_research_validation_workflow(
    manifest: dict[str, Any], *, manifest_sha256: str, base_dir: Path
) -> dict[str, Any]:
    """Validate artifact integrity, coverage, upstream outcomes, and safety declarations."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _valid_sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifacts_must_be_nonempty_list")
    required = manifest.get("required_audits", list(REQUIRED_AUDITS))
    if (not isinstance(required, list) or not required
            or any(not isinstance(name, str) or not name for name in required)):
        raise ValueError("required_audits_invalid")
    if len(set(required)) != len(required):
        raise ValueError("required_audits_must_be_unique")

    root = base_dir.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    dataset_hashes: dict[str, list[str]] = {}

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"code": "artifact_entry_invalid", "index": index})
            continue
        name = entry.get("audit_name")
        relpath = entry.get("path")
        item_errors: list[str] = []
        if not isinstance(name, str) or not name:
            name = f"unknown_{index}"
            item_errors.append("audit_name_invalid")
        if not isinstance(relpath, str) or not relpath:
            relpath = ""
            item_errors.append("artifact_path_invalid")
        if name in seen_names:
            item_errors.append("duplicate_audit_name")
        seen_names.add(name)
        if relpath in seen_paths:
            item_errors.append("duplicate_artifact_path")
        if relpath:
            seen_paths.add(relpath)

        payload = None
        digest = None
        if relpath:
            candidate = Path(relpath)
            if candidate.is_absolute() or ".." in candidate.parts:
                item_errors.append("artifact_path_must_be_relative_and_confined")
            else:
                resolved = (root / candidate).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    item_errors.append("artifact_path_escapes_manifest_directory")
                else:
                    try:
                        raw = resolved.read_bytes()
                        digest = _sha256(raw)
                        expected = entry.get("expected_report_sha256")
                        if expected is not None:
                            if not _valid_sha256(expected):
                                item_errors.append("expected_report_sha256_invalid")
                            elif digest != expected.lower():
                                item_errors.append("report_sha256_mismatch")
                        decoded = json.loads(raw.decode("utf-8"))
                        if not isinstance(decoded, dict):
                            item_errors.append("report_root_not_object")
                        else:
                            payload = decoded
                    except OSError:
                        item_errors.append("artifact_unreadable")
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        item_errors.append("artifact_json_invalid")

        upstream_status = None
        if payload is not None:
            if payload.get("audit_name") != name:
                item_errors.append("audit_name_mismatch")
            upstream_status = payload.get("status")
            if upstream_status not in KNOWN_STATUSES:
                item_errors.append("status_unknown_or_missing")
            elif upstream_status in {"FAIL", "NOT_COMPARABLE"}:
                item_errors.append("upstream_audit_failed" if upstream_status == "FAIL"
                                   else "upstream_audit_not_comparable")
            elif upstream_status == "PASS_WITH_WARNINGS":
                warnings.append(f"upstream_audit_passed_with_warnings:{name}")
            upstream_errors = payload.get("errors")
            if not isinstance(upstream_errors, list):
                item_errors.append("upstream_errors_field_missing_or_invalid")
            elif upstream_errors:
                item_errors.append("upstream_report_contains_errors")
            upstream_warnings = payload.get("warnings")
            if not isinstance(upstream_warnings, list):
                item_errors.append("upstream_warnings_field_missing_or_invalid")
            elif upstream_warnings and upstream_status == "PASS":
                warnings.append(f"upstream_warnings_present:{name}")
            for flag in FALSE_FLAGS:
                if payload.get(flag) is not False:
                    item_errors.append(f"safety_flag_not_explicitly_false:{flag}")
            for flag in OPTIONAL_FALSE_FLAGS:
                if flag in payload and payload[flag] is not False:
                    item_errors.append(f"safety_flag_not_false:{flag}")

            dataset_hash = payload.get("dataset_sha256")
            if dataset_hash is not None:
                if not _valid_sha256(dataset_hash):
                    item_errors.append("dataset_sha256_invalid")
                else:
                    dataset_hashes.setdefault(dataset_hash.lower(), []).append(name)
            elif name not in {
                "research_cohort_comparability_and_distribution_shift",
                "research_artifact_bundle_certification",
            }:
                warnings.append(f"dataset_fingerprint_not_exposed:{name}")

            if name == "feature_temporal_integrity":
                for field, code in (
                    (("prefix_invariance", "all_passed"), "prefix_invariance_not_confirmed"),
                    (("future_perturbation_invariance", "all_passed"), "future_perturbation_invariance_not_confirmed"),
                    (("chronological_partition_isolation", "purge_gaps_sufficient_for_horizon"), "temporal_split_purge_not_confirmed"),
                ):
                    nested = payload.get(field[0])
                    if not isinstance(nested, dict) or nested.get(field[1]) is not True:
                        item_errors.append(code)
            elif name == "feature_backend_parity":
                if payload.get("comparison_performed") is not True or payload.get("all_features_passed") is not True:
                    item_errors.append("backend_parity_not_confirmed")
            elif name == "pipeline_leakage_boundary":
                if not isinstance(payload.get("split_results"), dict) or "train" not in payload["split_results"]:
                    item_errors.append("pipeline_split_evidence_missing")
            elif name == "research_artifact_bundle_certification":
                if payload.get("missing_required_audits") != []:
                    item_errors.append("bundle_required_audit_coverage_incomplete")
                if not isinstance(payload.get("artifacts_checked"), int) or payload["artifacts_checked"] < 1:
                    item_errors.append("bundle_artifact_count_invalid")
            elif name == "walk_forward_robustness":
                supplied, validated = payload.get("evaluations_supplied"), payload.get("evaluations_validated")
                if not isinstance(supplied, int) or not isinstance(validated, int) or supplied < 1 or validated != supplied:
                    item_errors.append("walk_forward_evaluations_incomplete")
                seed_count = payload.get("unique_seed_count")
                if not isinstance(seed_count, int) or seed_count < 1:
                    item_errors.append("walk_forward_seed_coverage_missing")
                if not payload.get("observed_folds"):
                    item_errors.append("walk_forward_fold_coverage_missing")
            elif name == "research_cohort_comparability_and_distribution_shift":
                supplied, validated = payload.get("runs_supplied"), payload.get("runs_validated")
                if not isinstance(supplied, int) or not isinstance(validated, int) or supplied < 1 or validated != supplied:
                    item_errors.append("cohort_runs_incomplete")
                if not isinstance(payload.get("cohort_definition"), dict) or not payload["cohort_definition"]:
                    item_errors.append("cohort_definition_missing")
            elif name == "backtest_result_integrity":
                supplied, validated = payload.get("trades_supplied"), payload.get("trades_validated")
                if not isinstance(supplied, int) or not isinstance(validated, int) or supplied < 1 or validated != supplied:
                    item_errors.append("backtest_trade_ledger_incomplete")
                if not isinstance(payload.get("net_pnl_recomputed_from_valid_trades"), (int, float)):
                    item_errors.append("backtest_net_pnl_reconciliation_missing")

        for code in item_errors:
            errors.append({"code": code, "audit_name": name, "path": relpath})
        results.append({
            "audit_name": name, "path": relpath, "report_sha256": digest,
            "upstream_status": upstream_status, "passed": not item_errors,
            "errors": item_errors,
        })

    missing = sorted(set(required) - seen_names)
    for name in missing:
        errors.append({"code": "required_audit_missing", "audit_name": name})
    if len(dataset_hashes) > 1:
        errors.append({
            "code": "dataset_fingerprint_mismatch_across_reports",
            "dataset_hashes": {digest: names for digest, names in sorted(dataset_hashes.items())},
        })
    if not dataset_hashes:
        warnings.append("no_dataset_fingerprint_available_for_cross_report_reconciliation")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_validation_workflow",
        "manifest_sha256": manifest_sha256.lower(),
        "required_audits": required,
        "missing_required_audits": missing,
        "artifacts_checked": len(results),
        "artifact_results": results,
        "dataset_fingerprints": {digest: names for digest, names in sorted(dataset_hashes.items())},
        "errors": errors,
        "warnings": warnings,
        "status": status,
        "readiness": "NOT_READY" if status == "FAIL" else (
            "READY_WITH_WARNINGS" if status == "PASS_WITH_WARNINGS" else "READY"
        ),
        "actionable": False,
        "inferential": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "Validates supplied reports and declared evidence only; it does not run upstream audits or inspect raw market data.",
            "Dataset hashes are compared only when upstream reports expose them; absence is reported as a warning.",
            "A passing result is not proof of profitability, predictive validity, absence of all leakage, or deployment readiness.",
            "This workflow never authorizes model deployment or trading execution.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON validation manifest")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        report = audit_research_validation_workflow(
            manifest, manifest_sha256=_sha256(raw), base_dir=args.input.resolve().parent
        )
        output = args.output or args.input.with_name(args.input.stem + "_validation_workflow.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"], "readiness": report["readiness"],
            "report_path": str(output.resolve()), "artifacts_checked": report["artifacts_checked"],
            "error_count": len(report["errors"]), "warning_count": len(report["warnings"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

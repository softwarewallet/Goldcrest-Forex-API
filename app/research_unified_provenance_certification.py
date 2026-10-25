"""Unified offline research certification with provenance and freshness verification.

Consumes existing readiness, evidence-bundle, and provenance-manifest inputs; verifies their
contents against current files and emits one consolidated report. Never runs experiments,
regenerates artifacts, trains models, connects to a broker, or authorizes deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.research_artifact_freshness_audit import audit_artifact_freshness, fingerprint_path
from app.research_artifact_inventory_completeness import audit_artifact_inventory_completeness
from app.research_artifact_inventory_spec_seal import verify_inventory_spec_seal
from app.research_artifact_provenance import verify_provenance
from app.research_freshness_certification_gate import gate_freshness_certification
from app.research_git_source_provenance import inspect_git_provenance
from app.research_provenance_manifest_seal import verify_manifest_seal
from app.research_run_final_certification import run_final_certification

FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def run_unified_certification(
    readiness: dict[str, Any],
    bundle: dict[str, Any],
    provenance_manifest: dict[str, Any],
    *,
    provenance_seal: dict[str, Any] | None = None,
    inventory_spec: dict[str, Any] | None = None,
    inventory_seal: dict[str, Any] | None = None,
    evidence_root: Path,
    artifact_root: Path,
    current_source_revision: str | None = None,
    repo_path: Path | None = None,
    config_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    git_report: dict[str, Any] | None = None
    git_error: str | None = None
    if repo_path is None:
        git_error = "ValueError:repo_path_required_for_git_source_provenance"
    else:
        try:
            git_report = inspect_git_provenance(repo_path)
            resolved_revision = git_report.get("source_revision")
            if not isinstance(resolved_revision, str) or len(resolved_revision) != 40 or any(char not in "0123456789abcdef" for char in resolved_revision):
                raise ValueError("git_source_revision_invalid")
            current_source_revision = resolved_revision
            if git_report.get("status") != "PASS" or git_report.get("working_tree_clean") is not True:
                # Preserve the report for diagnostics, but never treat a dirty or internally
                # inconsistent Git result as a valid source identity.
                git_error = "ValueError:git_source_tree_not_clean"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            git_error = f"{type(exc).__name__}:{exc}"
    if current_source_revision is None:
        # Diagnostic placeholder only; git_error below guarantees this cannot certify.
        current_source_revision = "0" * 40
    seal_report = verify_manifest_seal(provenance_manifest, provenance_seal) if provenance_seal is not None else {
        "audit_name": "research_provenance_manifest_seal_verification",
        "status": "FAIL",
        "manifest_sha256": None,
        "errors": [{"code": "PROVENANCE_SEAL_REQUIRED"}],
    }
    inventory_report = audit_artifact_inventory_completeness(provenance_manifest, inventory_spec) if inventory_spec is not None else {
        "audit_name": "research_artifact_inventory_completeness_audit",
        "status": "FAIL",
        "errors": [{"code": "INDEPENDENT_INVENTORY_SPEC_REQUIRED"}],
        "manifest_artifact_count": None,
        "expected_artifact_count": None,
        "missing_or_mislocated_artifacts": [],
        "unexpected_or_mislocated_artifacts": [],
    }
    inventory_seal_report = verify_inventory_spec_seal(inventory_spec, inventory_seal) if inventory_spec is not None and inventory_seal is not None else {
        "audit_name": "research_artifact_inventory_specification_seal_verification",
        "status": "FAIL",
        "inventory_spec_sha256": None,
        "errors": [{"code": "INVENTORY_SPEC_SEAL_REQUIRED"}],
    }
    provenance_report = verify_provenance(
        provenance_manifest,
        root=artifact_root,
        current_source_revision=current_source_revision,
        config_path=config_path,
        dataset_path=dataset_path,
    )

    captured_artifacts = provenance_manifest.get("artifacts", [])
    if not isinstance(captured_artifacts, list):
        captured_artifacts = []
    config_record = provenance_manifest.get("config", {})
    dataset_record = provenance_manifest.get("dataset", {})
    freshness_manifest = {
        "audit_name": "research_artifact_freshness_manifest",
        "artifacts": [
            {
                "artifact_id": item.get("artifact_id"),
                "path": item.get("path"),
                "source_revision": item.get("source_revision"),
                "config_sha256": config_record.get("sha256") if isinstance(config_record, dict) else None,
                "dataset_sha256": dataset_record.get("sha256") if isinstance(dataset_record, dict) else None,
                "depends_on": item.get("depends_on", []),
            }
            for item in captured_artifacts if isinstance(item, dict)
        ],
    }
    freshness_report = audit_artifact_freshness(
        freshness_manifest,
        current_revision=current_source_revision,
        current_config_sha256=fingerprint_path(config_path),
        current_dataset_sha256=fingerprint_path(dataset_path),
    )
    bundle_verification, certification_report = run_final_certification(
        readiness, bundle, evidence_root=evidence_root
    )
    gate_report = gate_freshness_certification(freshness_report, certification_report)

    errors: list[dict[str, Any]] = []
    if git_error is not None or (git_report is not None and git_report.get("status") != "PASS"):
        errors.append({
            "code": "GIT_SOURCE_PROVENANCE_FAILED",
            "git_status": git_report.get("status") if git_report else "ERROR",
            "dirty_path_count": git_report.get("dirty_path_count") if git_report else None,
            "error": git_error,
        })
    if inventory_seal_report["status"] != "PASS":
        errors.append({
            "code": "INVENTORY_SPEC_SEAL_FAILED",
            "error_count": len(inventory_seal_report["errors"]),
        })
    if inventory_report["status"] != "PASS":
        errors.append({
            "code": "ARTIFACT_INVENTORY_COMPLETENESS_FAILED",
            "error_count": len(inventory_report["errors"]),
            "missing_count": len(inventory_report.get("missing_or_mislocated_artifacts", [])),
            "unexpected_count": len(inventory_report.get("unexpected_or_mislocated_artifacts", [])),
        })
    if seal_report["status"] != "PASS":
        errors.append({
            "code": "PROVENANCE_MANIFEST_SEAL_FAILED",
            "error_count": len(seal_report["errors"]),
        })
    if provenance_report["status"] != "PASS":
        errors.append({
            "code": "PROVENANCE_VERIFICATION_FAILED",
            "stale_or_invalid_count": provenance_report["stale_or_invalid_count"],
            "error_count": len(provenance_report["errors"]),
        })
    if gate_report["status"] not in {"PASS", "PASS_WITH_WARNINGS"}:
        errors.append({
            "code": "FRESHNESS_CERTIFICATION_GATE_FAILED",
            "gate_status": gate_report["status"],
            "error_count": len(gate_report["errors"]),
        })
    warnings = list(gate_report.get("warnings", []))
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_unified_provenance_certification",
        "status": status,
        "readiness": (
            "NOT_READY" if errors else
            "CERTIFIED_CURRENT_WITH_WARNINGS" if warnings else
            "CERTIFIED_CURRENT"
        ),
        "source_revision": current_source_revision,
        "git_source_provenance": git_report or {
            "status": "NOT_CHECKED",
            "errors": ([{"code": "GIT_SOURCE_PROVENANCE_ERROR", "error": git_error}] if git_error else []),
        },
        "artifact_inventory_spec_seal_verification": inventory_seal_report,
        "artifact_inventory_completeness": inventory_report,
        "provenance_manifest_seal_verification": seal_report,
        "provenance_verification": provenance_report,
        "bundle_verification": bundle_verification,
        "final_certification_chain": certification_report,
        "freshness_audit": freshness_report,
        "freshness_certification_gate": gate_report,
        "artifact_count": provenance_report.get("artifact_count"),
        "current_artifact_count": provenance_report.get("current_count"),
        "stale_or_invalid_artifact_count": provenance_report.get("stale_or_invalid_count"),
        "regeneration_required_count": freshness_report.get("regeneration_required_count"),
        "errors": errors,
        "warnings": sorted(set(warnings)),
        **FALSE_FLAGS,
        "scope_limitations": [
            "Inventory completeness is checked against the separately supplied inventory specification; a specification copied from the manifest does not independently prove completeness.",
            "When --repo-path is supplied, the source revision is resolved from Git HEAD and dirty tracked/untracked files fail certification; ignored files are not included by Git status.",
            "The detached provenance seal detects manifest edits only if the seal is retained separately or its digest is anchored in a separately controlled location; it is not a digital signature.",
            "Passing certification does not establish profitability, predictive validity, statistical significance, or deployment suitability.",
            "No artifacts are regenerated and no experiments, model training, broker connectivity, deployment, or trades are initiated.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--evidence-bundle", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--provenance-manifest", required=True, type=Path)
    parser.add_argument("--provenance-seal", required=True, type=Path, help="Detached seal created by app.research_provenance_manifest_seal")
    parser.add_argument("--inventory-spec", required=True, type=Path, help="Independently curated research artifact inventory contract")
    parser.add_argument("--inventory-seal", required=True, type=Path, help="Detached seal created by app.research_artifact_inventory_spec_seal")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo-path", required=True, type=Path, help="Local Git checkout; certification requires a clean working tree")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_unified_certification(
            _read_object(args.readiness_report),
            _read_object(args.evidence_bundle),
            _read_object(args.provenance_manifest),
            provenance_seal=_read_object(args.provenance_seal),
            inventory_spec=_read_object(args.inventory_spec),
            inventory_seal=_read_object(args.inventory_seal),
            evidence_root=args.evidence_root,
            artifact_root=args.artifact_root,
            repo_path=args.repo_path,
            config_path=args.config_path,
            dataset_path=args.dataset_path,
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "artifact_count": report["artifact_count"],
            "stale_or_invalid_artifact_count": report["stale_or_invalid_artifact_count"],
            "provenance_seal_status": report["provenance_manifest_seal_verification"]["status"],
            "inventory_completeness_status": report["artifact_inventory_completeness"]["status"],
            "inventory_seal_status": report["artifact_inventory_spec_seal_verification"]["status"],
            "regeneration_required_count": report["regeneration_required_count"],
            "error_count": len(report["errors"]),
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

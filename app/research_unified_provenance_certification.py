"""Unified offline research certification with provenance and freshness verification.

Consumes existing readiness, evidence-bundle, and provenance-manifest inputs; verifies their
contents against current files and emits one consolidated report. Never runs experiments,
regenerates artifacts, trains models, connects to a broker, or authorizes deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

from app.research_artifact_freshness_audit import audit_artifact_freshness, fingerprint_path
from app.research_artifact_inventory_completeness import audit_artifact_inventory_completeness
from app.research_artifact_inventory_spec_seal import verify_inventory_spec_seal
from app.research_artifact_provenance import verify_provenance
from app.research_freshness_certification_gate import gate_freshness_certification
from app.research_git_source_provenance import inspect_git_provenance
from app.research_provenance_manifest_seal import verify_manifest_seal
from app.research_certification_report_seal import create_report_seal
from app.research_run_final_certification import run_final_certification

def _normalize_component_report(name: str, report: Any) -> dict[str, Any]:
    """Fail closed when a verifier returns a malformed or internally inconsistent report."""
    if not isinstance(report, dict):
        return {
            "audit_name": name,
            "status": "FAIL",
            "errors": [{"code": "COMPONENT_REPORT_CONTRACT_INVALID",
                        "component": name, "reason": "report_must_be_object"}],
            "component_report_rejected": True,
        }

    status = report.get("status")
    errors = report.get("errors")
    warnings = report.get("warnings", [])
    reasons: list[str] = []
    if not isinstance(status, str) or not status.strip():
        reasons.append("status_must_be_nonempty_string")
    if not isinstance(errors, list) or any(not isinstance(item, dict) for item in errors):
        reasons.append("errors_must_be_list_of_objects")
    elif status == "PASS" and errors:
        reasons.append("pass_status_must_not_contain_errors")
    if not isinstance(warnings, list) or any(not isinstance(item, str) or not item.strip() for item in warnings):
        reasons.append("warnings_must_be_list_of_nonempty_strings")
    elif status == "PASS" and warnings:
        reasons.append("pass_with_warnings_must_use_pass_with_warnings_status")

    if reasons:
        return {
            "audit_name": name,
            "status": "FAIL",
            "errors": [{
                "code": "COMPONENT_REPORT_CONTRACT_INVALID",
                "component": name,
                "reasons": reasons,
            }],
            "component_report_rejected": True,
        }
    return report


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
    if not isinstance(readiness, dict):
        readiness = {}
    if not isinstance(bundle, dict):
        bundle = {}
    if not isinstance(provenance_manifest, dict):
        provenance_manifest = {}

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
    try:
        seal_report = verify_manifest_seal(provenance_manifest, provenance_seal) if provenance_seal is not None else {
            "audit_name": "research_provenance_manifest_seal_verification",
            "status": "FAIL",
            "manifest_sha256": None,
            "errors": [{"code": "PROVENANCE_SEAL_REQUIRED"}],
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        seal_report = {
            "audit_name": "research_provenance_manifest_seal_verification",
            "status": "FAIL",
            "manifest_sha256": None,
            "errors": [{"code": "PROVENANCE_SEAL_VERIFICATION_EXECUTION_FAILED",
                        "error": f"{type(exc).__name__}:{exc}"}],
        }
    try:
        inventory_report = audit_artifact_inventory_completeness(provenance_manifest, inventory_spec) if inventory_spec is not None else {
            "audit_name": "research_artifact_inventory_completeness_audit",
            "status": "FAIL",
            "errors": [{"code": "INDEPENDENT_INVENTORY_SPEC_REQUIRED"}],
            "manifest_artifact_count": None,
            "expected_artifact_count": None,
            "missing_or_mislocated_artifacts": [],
            "unexpected_or_mislocated_artifacts": [],
        }
    except (OSError, ValueError, TypeError) as exc:
        inventory_report = {
            "audit_name": "research_artifact_inventory_completeness_audit",
            "status": "FAIL",
            "errors": [{"code": "INVENTORY_AUDIT_EXECUTION_FAILED", "error": f"{type(exc).__name__}:{exc}"}],
            "manifest_artifact_count": None,
            "expected_artifact_count": None,
            "missing_or_mislocated_artifacts": [],
            "unexpected_or_mislocated_artifacts": [],
        }
    try:
        inventory_seal_report = verify_inventory_spec_seal(inventory_spec, inventory_seal) if inventory_spec is not None and inventory_seal is not None else {
            "audit_name": "research_artifact_inventory_specification_seal_verification",
            "status": "FAIL",
            "inventory_spec_sha256": None,
            "errors": [{"code": "INVENTORY_SPEC_SEAL_REQUIRED"}],
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        inventory_seal_report = {
            "audit_name": "research_artifact_inventory_specification_seal_verification",
            "status": "FAIL",
            "inventory_spec_sha256": None,
            "errors": [{"code": "INVENTORY_SPEC_SEAL_VERIFICATION_EXECUTION_FAILED",
                        "error": f"{type(exc).__name__}:{exc}"}],
        }
    try:
        provenance_report = verify_provenance(
            provenance_manifest,
            root=artifact_root,
            current_source_revision=current_source_revision,
            config_path=config_path,
            dataset_path=dataset_path,
        )
    except (OSError, ValueError, TypeError) as exc:
        provenance_report = {
            "audit_name": "research_artifact_provenance_verification",
            "status": "FAIL",
            "artifact_count": 0,
            "current_count": 0,
            "stale_or_invalid_count": 0,
            "artifacts": [],
            "errors": [{"code": "PROVENANCE_AUDIT_EXECUTION_FAILED", "error": f"{type(exc).__name__}:{exc}"}],
        }

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
    fingerprint_errors: list[dict[str, Any]] = []

    def _safe_fingerprint(label: str, path: Path) -> str:
        try:
            return fingerprint_path(path)
        except (OSError, ValueError) as exc:
            fingerprint_errors.append({
                "code": "CURRENT_INPUT_FINGERPRINT_FAILED",
                "input": label,
                "path": str(path),
                "error": f"{type(exc).__name__}:{exc}",
            })
            # Invalid placeholder forces the freshness validator to reject the input too;
            # the explicit error below is the authoritative fail-closed signal.
            return ""

    current_config_sha256 = _safe_fingerprint("config", config_path)
    current_dataset_sha256 = _safe_fingerprint("dataset", dataset_path)
    try:
        freshness_report = audit_artifact_freshness(
            freshness_manifest,
            current_revision=current_source_revision,
            current_config_sha256=current_config_sha256,
            current_dataset_sha256=current_dataset_sha256,
        )
    except (OSError, ValueError, TypeError) as exc:
        freshness_report = {
            "audit_name": "research_artifact_freshness_audit",
            "status": "FAIL",
            "readiness": "REGENERATION_REQUIRED",
            "artifact_count": len(freshness_manifest["artifacts"]),
            "current_count": 0,
            "regeneration_required_count": len(freshness_manifest["artifacts"]),
            "artifacts": [],
            "regeneration_order": [],
            "errors": [{"code": "FRESHNESS_AUDIT_EXECUTION_FAILED", "error": f"{type(exc).__name__}:{exc}"}],
        }
    try:
        bundle_verification, certification_report = run_final_certification(
            readiness, bundle, evidence_root=evidence_root
        )
    except (OSError, ValueError, TypeError) as exc:
        failure = {"code": "FINAL_CERTIFICATION_EXECUTION_FAILED", "error": f"{type(exc).__name__}:{exc}"}
        bundle_verification = {
            "audit_name": "research_evaluation_certification_evidence_bundle_verification",
            "status": "FAIL",
            "errors": [failure],
        }
        certification_report = {
            "audit_name": "research_final_certification_chain",
            "status": "FAIL",
            "readiness": "NOT_CERTIFIED",
            "errors": [failure],
        }
    try:
        gate_report = gate_freshness_certification(freshness_report, certification_report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        gate_report = {
            "audit_name": "research_freshness_certification_gate",
            "status": "FAIL",
            "errors": [{"code": "FRESHNESS_CERTIFICATION_GATE_EXECUTION_FAILED",
                        "error": f"{type(exc).__name__}:{exc}"}],
            "warnings": [],
        }

    # Verifier status and its structured diagnostics must agree. A component that
    # claims PASS while reporting errors cannot be allowed to certify the run.
    component_reports = (
        ("artifact_inventory_spec_seal_verification", inventory_seal_report),
        ("artifact_inventory_completeness", inventory_report),
        ("provenance_manifest_seal_verification", seal_report),
        ("provenance_verification", provenance_report),
        ("bundle_verification", bundle_verification),
        ("final_certification_chain", certification_report),
        ("freshness_audit", freshness_report),
        ("freshness_certification_gate", gate_report),
    )
    for component_name, component_report in component_reports:
        normalized = _normalize_component_report(component_name, component_report)
        if normalized is not component_report:
            if component_name == "artifact_inventory_spec_seal_verification":
                inventory_seal_report = normalized
            elif component_name == "artifact_inventory_completeness":
                inventory_report = normalized
            elif component_name == "provenance_manifest_seal_verification":
                seal_report = normalized
            elif component_name == "provenance_verification":
                provenance_report = normalized
            elif component_name == "bundle_verification":
                bundle_verification = normalized
            elif component_name == "final_certification_chain":
                certification_report = normalized
            elif component_name == "freshness_audit":
                freshness_report = normalized
            elif component_name == "freshness_certification_gate":
                gate_report = normalized

    errors: list[dict[str, Any]] = list(fingerprint_errors)
    for component_name, component_report in component_reports:
        # Use the normalized bindings above, rather than the original untrusted object.
        current_report = {
            "artifact_inventory_spec_seal_verification": inventory_seal_report,
            "artifact_inventory_completeness": inventory_report,
            "provenance_manifest_seal_verification": seal_report,
            "provenance_verification": provenance_report,
            "bundle_verification": bundle_verification,
            "final_certification_chain": certification_report,
            "freshness_audit": freshness_report,
            "freshness_certification_gate": gate_report,
        }[component_name]
        if current_report.get("component_report_rejected") is True:
            errors.append({
                "code": "CERTIFICATION_COMPONENT_REPORT_INVALID",
                "component": component_name,
                "reasons": current_report["errors"][0].get("reasons", [
                    current_report["errors"][0].get("reason", "invalid_report")
                ]),
            })
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
    if provenance_report.get("status") != "PASS":
        errors.append({
            "code": "PROVENANCE_VERIFICATION_FAILED",
            "stale_or_invalid_count": provenance_report.get("stale_or_invalid_count"),
            "error_count": len(provenance_report.get("errors", [])),
        })
    if bundle_verification.get("status") != "PASS":
        errors.append({
            "code": "EVIDENCE_BUNDLE_VERIFICATION_FAILED",
            "error_count": len(bundle_verification.get("errors", [])),
        })
    if certification_report.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        errors.append({
            "code": "FINAL_CERTIFICATION_CHAIN_FAILED",
            "certification_status": certification_report.get("status"),
            "error_count": len(certification_report.get("errors", [])),
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
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload



def _validate_cli_paths(args: argparse.Namespace) -> None:
    """Reject output aliases before creating directories or writing any files."""
    inputs = {
        "readiness report": args.readiness_report,
        "evidence bundle": args.evidence_bundle,
        "provenance manifest": args.provenance_manifest,
        "provenance seal": args.provenance_seal,
        "inventory specification": args.inventory_spec,
        "inventory seal": args.inventory_seal,
        "config": args.config_path,
        "dataset": args.dataset_path,
    }
    outputs = {"report output": args.output}
    if args.report_seal_output is not None:
        outputs["report seal output"] = args.report_seal_output
    resolved_inputs = {label: path.resolve(strict=False) for label, path in inputs.items()}
    resolved_outputs = {label: path.resolve(strict=False) for label, path in outputs.items()}
    for output_label, output_path in resolved_outputs.items():
        for input_label, input_path in resolved_inputs.items():
            if output_path == input_path:
                raise ValueError(f"output_path_aliases_input:{output_label}:{input_label}:{output_path}")
    items = list(resolved_outputs.items())
    for index, (label, path) in enumerate(items):
        for other_label, other_path in items[index + 1:]:
            if path == other_path:
                raise ValueError(f"output_paths_must_differ:{label}:{other_label}:{path}")
    for label, path in outputs.items():
        if path.exists():
            raise ValueError(f"output_already_exists:{label}:{path}")



def _validate_input_paths(args: argparse.Namespace) -> None:
    """Validate required input existence and filesystem types before audit work begins."""
    json_inputs = {
        "readiness report": args.readiness_report,
        "evidence bundle": args.evidence_bundle,
        "provenance manifest": args.provenance_manifest,
        "provenance seal": args.provenance_seal,
        "inventory specification": args.inventory_spec,
        "inventory seal": args.inventory_seal,
    }
    for label, path in json_inputs.items():
        if not path.exists():
            raise ValueError(f"required_input_missing:{label}:{path}")
        if not path.is_file():
            raise ValueError(f"required_input_not_file:{label}:{path}")
    directory_inputs = {
        "evidence root": args.evidence_root,
        "artifact root": args.artifact_root,
        "repository path": args.repo_path,
    }
    for label, path in directory_inputs.items():
        if not path.exists():
            raise ValueError(f"required_input_missing:{label}:{path}")
        if not path.is_dir():
            raise ValueError(f"required_input_not_directory:{label}:{path}")
    for label, path in (("config", args.config_path), ("dataset", args.dataset_path)):
        if not path.exists():
            raise ValueError(f"required_input_missing:{label}:{path}")
        if not (path.is_file() or path.is_dir()):
            raise ValueError(f"required_input_not_file_or_directory:{label}:{path}")


def _write_json_outputs(report_path: Path, report: dict[str, Any],
                        seal_path: Path | None = None,
                        seal: dict[str, Any] | None = None) -> None:
    """Stage JSON outputs before publication and remove partial outputs on failure."""
    payloads: list[tuple[Path, str]] = [
        (report_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    ]
    if seal_path is not None and seal is not None:
        payloads.append((seal_path, json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"))
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for target, payload in payloads:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((Path(handle.name), target))
        for temporary, target in staged:
            # Publish without clobbering a target created by another process after
            # staging. A same-directory hard link is atomic and fails if target exists.
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise ValueError(f"output_already_exists:{target}") from None
            # Record the published inode so rollback never removes a different
            # file if another process replaces this path before cleanup.
            stat = target.stat()
            published.append((target, (stat.st_dev, stat.st_ino)))
            temporary.unlink()
    except Exception:
        for target, published_identity in published:
            try:
                current = target.stat()
                if (current.st_dev, current.st_ino) == published_identity:
                    target.unlink()
            except OSError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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
    parser.add_argument("--report-seal-output", type=Path, help="Optional detached integrity seal for the consolidated report")
    args = parser.parse_args()
    try:
        _validate_cli_paths(args)
        _validate_input_paths(args)
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
        seal = create_report_seal(report) if args.report_seal_output is not None else None
        _write_json_outputs(args.output, report, args.report_seal_output, seal)
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
            "report_seal_path": str(args.report_seal_output.resolve()) if seal is not None and args.report_seal_output is not None else None,
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

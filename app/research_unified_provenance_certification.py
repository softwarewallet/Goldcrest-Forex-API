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
from app.research_artifact_provenance import verify_provenance
from app.research_freshness_certification_gate import gate_freshness_certification
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
    evidence_root: Path,
    artifact_root: Path,
    current_source_revision: str,
    config_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
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
            "This command validates supplied manifests and reports and checks declared artifact bytes; it cannot prove that the inventory is complete.",
            "The source revision is caller-supplied and syntax/consistency-checked, not independently discovered from the running Git checkout.",
            "The provenance manifest must be retained separately or its digest anchored in a separately controlled location.",
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
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--current-source-revision", required=True)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_unified_certification(
            _read_object(args.readiness_report),
            _read_object(args.evidence_bundle),
            _read_object(args.provenance_manifest),
            evidence_root=args.evidence_root,
            artifact_root=args.artifact_root,
            current_source_revision=args.current_source_revision,
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

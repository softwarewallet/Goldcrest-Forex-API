"""Audit evaluation-window alignment and concrete output evidence for planned research runs.

This is an offline integrity audit over a plan, a registry report, and the local manifest/artifact
files referenced by that report. It does not execute experiments, recalculate trading metrics,
train/select models, or connect to broker/trading systems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which are not valid RFC 8259 JSON."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result if result.tzinfo is not None else None


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit_evaluation_output_integrity(
    plan: dict[str, Any],
    registry_report: dict[str, Any],
    *,
    registry_root: Path,
    require_output_artifact: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not isinstance(plan, dict) or plan.get("audit_name") != "research_experiment_plan":
        raise ValueError("plan_report_invalid")
    if plan.get("status") != "PASS":
        raise ValueError("plan_report_not_passing")
    if not isinstance(registry_report, dict) or registry_report.get("audit_name") != "research_experiment_registry":
        raise ValueError("registry_report_invalid")
    if registry_report.get("status") not in PASS_STATUSES:
        raise ValueError("registry_report_not_passing")

    matrix = plan.get("run_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("plan_run_matrix_invalid")
    planned: dict[str, dict[str, Any]] = {}
    for item in matrix:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str) or not item["run_id"]:
            raise ValueError("plan_run_identity_invalid")
        if item["run_id"] in planned:
            raise ValueError("plan_run_id_duplicate")
        planned[item["run_id"]] = item

    run_results = registry_report.get("run_results")
    if not isinstance(run_results, list):
        raise ValueError("registry_run_results_invalid")
    root = registry_root.resolve()
    seen: set[str] = set()
    checked_runs = 0
    artifacts_checked = 0
    total_artifact_bytes = 0
    metric_values_checked = 0

    for index, record in enumerate(run_results):
        if not isinstance(record, dict):
            errors.append({"code": "registry_run_record_invalid", "index": index})
            continue
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            errors.append({"code": "registry_run_id_invalid", "index": index})
            continue
        if run_id in seen:
            errors.append({"code": "registry_run_id_duplicate", "run_id": run_id})
            continue
        seen.add(run_id)
        expected = planned.get(run_id)
        if expected is None:
            warnings.append(f"unplanned_run_not_window_checked:{run_id}")
            continue
        if record.get("status") != "PASS":
            errors.append({"code": "registry_run_not_passed", "run_id": run_id})
            continue
        manifest_path = record.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            errors.append({"code": "manifest_path_missing_from_registry_record", "run_id": run_id})
            continue
        relative = Path(manifest_path)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append({"code": "manifest_path_must_be_confined", "run_id": run_id})
            continue
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append({"code": "manifest_path_escapes_registry_root", "run_id": run_id})
            continue
        try:
            raw = resolved.read_bytes()
            manifest = _load_json_object(raw, source=str(resolved))
        except OSError:
            errors.append({"code": "manifest_unreadable", "run_id": run_id})
            continue
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            errors.append({"code": "manifest_json_invalid", "run_id": run_id})
            continue
        if not isinstance(manifest, dict):
            errors.append({"code": "manifest_root_must_be_object", "run_id": run_id})
            continue
        if manifest.get("run_id") != run_id:
            errors.append({"code": "manifest_run_id_mismatch", "run_id": run_id})
        if record.get("manifest_sha256") != _sha256(raw):
            errors.append({"code": "manifest_sha256_differs_from_registry_report", "run_id": run_id})

        expected_window = expected.get("evaluation_window")
        actual_window = manifest.get("evaluation_window")
        if not isinstance(expected_window, dict) or not isinstance(actual_window, dict):
            errors.append({"code": "evaluation_window_missing", "run_id": run_id})
        else:
            start = _timestamp(actual_window.get("start"))
            end = _timestamp(actual_window.get("end"))
            expected_start = _timestamp(expected_window.get("start"))
            expected_end = _timestamp(expected_window.get("end"))
            if any(x is None for x in (start, end, expected_start, expected_end)):
                errors.append({"code": "evaluation_window_timestamp_invalid", "run_id": run_id})
            else:
                if start.utcoffset().total_seconds() != 0 or end.utcoffset().total_seconds() != 0:
                    errors.append({"code": "evaluation_window_not_utc", "run_id": run_id})
                if start >= end:
                    errors.append({"code": "evaluation_window_order_invalid", "run_id": run_id})
                if start != expected_start or end != expected_end:
                    errors.append({
                        "code": "evaluation_window_does_not_match_plan",
                        "run_id": run_id,
                        "planned": expected_window,
                        "observed": actual_window,
                    })

        metrics = manifest.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            errors.append({"code": "evaluation_metrics_missing", "run_id": run_id})
        else:
            for name, value in metrics.items():
                if not isinstance(name, str) or not name.strip() or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    errors.append({"code": "evaluation_metric_invalid", "run_id": run_id, "metric": str(name)})
                else:
                    metric_values_checked += 1

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            errors.append({"code": "artifact_list_invalid", "run_id": run_id})
            artifacts = []
        if require_output_artifact and not artifacts:
            errors.append({"code": "output_artifact_evidence_missing", "run_id": run_id})
        if artifacts:
            valid_artifacts_for_run = 0
            artifact_paths: set[str] = set()
            for artifact_index, artifact in enumerate(artifacts):
                if not isinstance(artifact, dict):
                    errors.append({"code": "artifact_record_invalid", "run_id": run_id, "index": artifact_index})
                    continue
                path_value = artifact.get("path")
                if not isinstance(path_value, str) or not path_value:
                    errors.append({"code": "artifact_path_invalid", "run_id": run_id, "index": artifact_index})
                    continue
                if path_value in artifact_paths:
                    errors.append({"code": "artifact_path_duplicate", "run_id": run_id, "path": path_value})
                    continue
                artifact_paths.add(path_value)
                artifact_relative = Path(path_value)
                if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
                    errors.append({"code": "artifact_path_must_be_confined", "run_id": run_id, "path": path_value})
                    continue
                artifact_path = (resolved.parent / artifact_relative).resolve()
                try:
                    artifact_path.relative_to(resolved.parent.resolve())
                except ValueError:
                    errors.append({"code": "artifact_path_escapes_manifest_root", "run_id": run_id, "path": path_value})
                    continue
                try:
                    artifact_bytes = artifact_path.read_bytes()
                except OSError:
                    errors.append({"code": "artifact_unreadable", "run_id": run_id, "path": path_value})
                    continue
                actual_hash = _sha256(artifact_bytes)
                expected_hash = artifact.get("sha256")
                if not isinstance(expected_hash, str) or len(expected_hash) != 64 or actual_hash != expected_hash.lower():
                    errors.append({"code": "artifact_sha256_mismatch_or_invalid", "run_id": run_id, "path": path_value})
                    continue
                if not artifact_bytes:
                    errors.append({"code": "artifact_empty", "run_id": run_id, "path": path_value})
                    continue
                valid_artifacts_for_run += 1
                artifacts_checked += 1
                total_artifact_bytes += len(artifact_bytes)
            if require_output_artifact and valid_artifacts_for_run == 0:
                errors.append({"code": "no_valid_output_artifact", "run_id": run_id})
        checked_runs += 1

    missing = sorted(set(planned) - seen)
    for run_id in missing:
        errors.append({"code": "planned_run_missing_from_registry", "run_id": run_id})

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_evaluation_output_integrity",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "registry_id": registry_report.get("registry_id"),
        "status": status,
        "readiness": "NOT_INTEGRITY_VERIFIED" if errors else (
            "OUTPUT_INTEGRITY_VERIFIED_WITH_WARNINGS" if warnings else "EVALUATION_WINDOWS_AND_OUTPUTS_VERIFIED"
        ),
        "planned_runs": len(planned),
        "registry_records": len(run_results),
        "planned_runs_checked": checked_runs,
        "missing_planned_runs": missing,
        "artifacts_checked": artifacts_checked,
        "artifact_bytes_checked": total_artifact_bytes,
        "finite_metric_values_checked": metric_values_checked,
        "require_output_artifact": require_output_artifact,
        "errors": errors,
        "warnings": warnings,
        **FALSE_FLAGS,
        "deployment_authorized": False,
        "scope_limitations": [
            "Checks manifest-declared evaluation windows against the plan and verifies declared artifact bytes against SHA-256 digests.",
            "Does not recompute metrics from raw market data or prove that metric calculations used only the declared window.",
            "A passing result establishes file integrity and metadata alignment only, not predictive validity or profitability.",
            "No experiments are executed and no trading or broker functionality is accessed.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registry-report", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path, help="Root directory used by the registry audit")
    parser.add_argument("--allow-no-output-artifacts", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_evaluation_output_integrity(
            _read_object(args.plan),
            _read_object(args.registry_report),
            registry_root=args.registry_root,
            require_output_artifact=not args.allow_no_output_artifacts,
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "planned_runs_checked": report["planned_runs_checked"],
            "artifacts_checked": report["artifacts_checked"],
            "finite_metric_values_checked": report["finite_metric_values_checked"],
            "error_count": len(report["errors"]),
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

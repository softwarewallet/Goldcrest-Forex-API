"""Offline experiment-manifest provenance and descriptive comparison audit.

This module validates supplied experiment metadata and artifact checksums. It does not
train models, execute experiments, infer significance, select winners, or connect to
broker/trading systems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


FALSE_FLAGS = (
    "actionable",
    "inferential",
    "statistical_significance_claimed",
    "automatic_model_selection",
    "model_training_performed",
    "execution_connected",
    "automatic_repair_performed",
)
REQUIRED_CONTEXT = ("symbol", "timeframe")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_experiment_manifest(
    manifest: dict[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    """Validate one experiment manifest and hash every declared output artifact."""
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")

    if manifest.get("schema_version") != 1:
        errors.append({"code": "schema_version_unsupported"})
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
        errors.append({"code": "run_id_invalid"})
    created_at = _timestamp(manifest.get("created_at_utc"))
    if created_at is None or created_at.utcoffset() is None:
        errors.append({"code": "created_at_utc_invalid"})
    elif created_at.utcoffset().total_seconds() != 0:
        errors.append({"code": "created_at_utc_must_be_utc"})

    context = manifest.get("context")
    if not isinstance(context, dict):
        errors.append({"code": "context_must_be_object"})
        context = {}
    for field in REQUIRED_CONTEXT:
        if not isinstance(context.get(field), str) or not context[field].strip() or context[field] != context[field].strip():
            errors.append({"code": f"context_{field}_invalid"})

    for field in ("config_sha256", "dataset_sha256"):
        if not valid_sha256(manifest.get(field)):
            errors.append({"code": f"{field}_invalid"})
    commit = manifest.get("code_commit_sha")
    if not isinstance(commit, str) or not (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
        errors.append({"code": "code_commit_sha_invalid"})
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append({"code": "seed_invalid"})
    fold = manifest.get("fold_id")
    if not isinstance(fold, (str, int)) or isinstance(fold, bool) or str(fold).strip() == "" or (isinstance(fold, str) and fold != fold.strip()):
        errors.append({"code": "fold_id_invalid"})

    window = manifest.get("evaluation_window")
    if not isinstance(window, dict):
        errors.append({"code": "evaluation_window_invalid"})
    else:
        start = _timestamp(window.get("start"))
        end = _timestamp(window.get("end"))
        if start is None or end is None:
            errors.append({"code": "evaluation_window_timestamps_invalid"})
        elif start >= end:
            errors.append({"code": "evaluation_window_order_invalid"})

    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append({"code": "metrics_must_be_nonempty_object"})
        metrics = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            errors.append({"code": "metric_name_invalid"})
        if not finite_number(value):
            errors.append({"code": f"metric_value_invalid:{name}"})

    flags = manifest.get("safety_flags")
    if not isinstance(flags, dict):
        errors.append({"code": "safety_flags_must_be_object"})
        flags = {}
    for name in FALSE_FLAGS:
        if flags.get(name) is not False:
            errors.append({"code": f"safety_flag_not_explicitly_false:{name}"})

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        errors.append({"code": "artifacts_must_be_list"})
        artifacts = []
    seen_paths: set[str] = set()
    checked_artifacts: list[dict[str, Any]] = []
    root = base_dir.resolve()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            errors.append({"code": "artifact_entry_invalid", "index": index})
            continue
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append({"code": "artifact_path_invalid", "index": index})
            continue
        if path_value in seen_paths:
            errors.append({"code": "artifact_path_duplicate", "path": path_value})
            continue
        seen_paths.add(path_value)
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append({"code": "artifact_path_must_be_confined", "path": path_value})
            continue
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append({"code": "artifact_path_escapes_root", "path": path_value})
            continue
        try:
            raw = resolved.read_bytes()
        except OSError:
            errors.append({"code": "artifact_unreadable", "path": path_value})
            continue
        actual = sha256_bytes(raw)
        expected = item.get("sha256")
        if not valid_sha256(expected):
            errors.append({"code": "artifact_sha256_invalid", "path": path_value})
        elif actual != expected.lower():
            errors.append({"code": "artifact_sha256_mismatch", "path": path_value})
        checked_artifacts.append({"path": path_value, "sha256": actual, "bytes": len(raw)})

    if not artifacts:
        warnings.append("no_output_artifacts_declared")
    return {
        "schema_version": 1,
        "audit_name": "research_experiment_provenance",
        "status": "PASS" if not errors else "FAIL",
        "readiness": "READY_FOR_DESCRIPTIVE_COMPARISON" if not errors else "NOT_READY",
        "run_id": run_id if isinstance(run_id, str) else None,
        "context": context,
        "config_sha256": manifest.get("config_sha256"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "code_commit_sha": commit,
        "seed": seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        "fold_id": str(fold) if isinstance(fold, (str, int)) and not isinstance(fold, bool) else None,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(k, str) and finite_number(v)},
        "artifacts_checked": checked_artifacts,
        "errors": errors,
        "warnings": warnings,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
    }


def compare_experiment_manifests(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe metric spread across validated runs; never rank/select a winner."""
    errors: list[dict[str, Any]] = []
    if not isinstance(reports, list) or len(reports) < 2:
        raise ValueError("at_least_two_reports_required")
    seen_ids: set[str] = set()
    valid_reports: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, dict) or report.get("status") != "PASS":
            errors.append({"code": "input_report_not_validated", "index": index})
            continue
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            errors.append({"code": "run_id_invalid", "index": index})
        elif run_id in seen_ids:
            errors.append({"code": "duplicate_run_id", "run_id": run_id})
        seen_ids.add(run_id if isinstance(run_id, str) else f"invalid_{index}")
        valid_reports.append(report)

    if valid_reports:
        base = valid_reports[0]
        for index, report in enumerate(valid_reports[1:], start=1):
            for field in ("context", "config_sha256", "dataset_sha256"):
                if report.get(field) != base.get(field):
                    errors.append({"code": f"comparison_{field}_mismatch", "index": index})
    metric_sets = [set(r.get("metrics", {})) for r in valid_reports]
    if metric_sets and any(names != metric_sets[0] for names in metric_sets[1:]):
        errors.append({"code": "comparison_metric_set_mismatch"})

    summaries: dict[str, Any] = {}
    metric_names = sorted(set.intersection(*metric_sets)) if metric_sets else []
    for name in metric_names:
        values = [r["metrics"][name] for r in valid_reports]
        if not all(finite_number(v) for v in values):
            errors.append({"code": f"comparison_metric_value_invalid:{name}"})
            continue
        summaries[name] = {
            "observations": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "population_standard_deviation": statistics.pstdev(values),
        }
    return {
        "schema_version": 1,
        "audit_name": "research_experiment_comparison",
        "status": "PASS" if not errors else "FAIL",
        "readiness": "DESCRIPTIVE_COMPARISON_ONLY" if not errors else "NOT_COMPARABLE",
        "runs_supplied": len(reports),
        "runs_compared": len(valid_reports),
        "metric_summaries": summaries,
        "errors": errors,
        "warnings": [],
        "selection_performed": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "actionable": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError("json_root_must_be_object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate one experiment manifest")
    validate.add_argument("manifest", type=Path)
    compare = sub.add_parser("compare", help="compare validated provenance reports")
    compare.add_argument("reports", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        raw = args.manifest.read_bytes()
        manifest = _load_json_object(raw, source=str(args.manifest))
        report = validate_experiment_manifest(manifest, base_dir=args.manifest.resolve().parent)
    else:
        report = compare_experiment_manifests([_read_json(path) for path in args.reports])
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if getattr(args, "output", None):
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

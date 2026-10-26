"""Build a deterministic offline experiment matrix from a declarative research plan.

The planner expands seed × fold × evaluation-window coverage and fingerprints the plan.
It never launches jobs, reads market data, trains/selects models, or touches execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

MAX_PLANNED_RUNS = 5000
FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unique_nonempty(values: Any, label: str) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}_must_be_nonempty_list")
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, (str, int)) or isinstance(item, bool) or not str(item).strip():
            raise ValueError(f"{label}_item_invalid")
        item_text = str(item)
        if item_text != item_text.strip():
            raise ValueError(f"{label}_item_not_canonical")
        normalized.append(item_text)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label}_must_be_unique")
    return values


def build_experiment_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a declarative plan and return a stable, explicit run matrix."""
    if not isinstance(plan, dict):
        raise ValueError("plan_must_be_object")
    if plan.get("schema_version") != 1:
        raise ValueError("schema_version_unsupported")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", plan_id):
        raise ValueError("plan_id_invalid")

    context = plan.get("context")
    if not isinstance(context, dict):
        raise ValueError("context_must_be_object")
    for field in ("symbol", "timeframe"):
        if (
            not isinstance(context.get(field), str)
            or not context[field].strip()
            or context[field] != context[field].strip()
        ):
            raise ValueError(f"context_{field}_invalid")
    for field in ("config_sha256", "dataset_sha256"):
        if not _valid_digest(plan.get(field)):
            raise ValueError(f"{field}_invalid")
    commit = plan.get("code_commit_sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("code_commit_sha_invalid")

    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError("seeds_must_be_nonempty_integer_list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds_must_be_unique")

    folds = _unique_nonempty(plan.get("folds"), "folds")
    windows = plan.get("evaluation_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("evaluation_windows_must_be_nonempty_list")
    normalized_windows: list[dict[str, str]] = []
    window_ids: set[str] = set()
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"evaluation_window_invalid:{index}")
        window_id = window.get("window_id")
        if (
            not isinstance(window_id, (str, int))
            or isinstance(window_id, bool)
            or not str(window_id).strip()
            or str(window_id) != str(window_id).strip()
        ):
            raise ValueError(f"evaluation_window_id_invalid:{index}")
        window_id = str(window_id)
        if window_id in window_ids:
            raise ValueError("evaluation_window_ids_must_be_unique")
        window_ids.add(window_id)
        start = _timestamp(window.get("start"))
        end = _timestamp(window.get("end"))
        if start is None or end is None:
            raise ValueError(f"evaluation_window_timestamps_invalid:{window_id}")
        if start.utcoffset().total_seconds() != 0 or end.utcoffset().total_seconds() != 0:
            raise ValueError(f"evaluation_window_must_be_utc:{window_id}")
        if start >= end:
            raise ValueError(f"evaluation_window_order_invalid:{window_id}")
        normalized_windows.append({
            "window_id": window_id,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        })

    run_count = len(seeds) * len(folds) * len(normalized_windows)
    if run_count > MAX_PLANNED_RUNS:
        raise ValueError(f"planned_run_limit_exceeded:{run_count}>{MAX_PLANNED_RUNS}")
    canonical_plan = _canonical_bytes(plan)
    plan_sha256 = _sha256(canonical_plan)

    run_matrix: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            for window in normalized_windows:
                identity = {
                    "plan_sha256": plan_sha256,
                    "seed": seed,
                    "fold_id": str(fold),
                    "window_id": window["window_id"],
                }
                run_id = "run-" + _sha256(_canonical_bytes(identity))[:20]
                run_matrix.append({
                    "run_id": run_id,
                    "seed": seed,
                    "fold_id": str(fold),
                    "window_id": window["window_id"],
                    "evaluation_window": {"start": window["start"], "end": window["end"]},
                    "context": dict(context),
                    "config_sha256": plan["config_sha256"].lower(),
                    "dataset_sha256": plan["dataset_sha256"].lower(),
                    "code_commit_sha": commit.lower(),
                    "plan_sha256": plan_sha256,
                    "execution_scheduled": False,
                })

    expected_run_ids = [item["run_id"] for item in run_matrix]
    return {
        "schema_version": 1,
        "audit_name": "research_experiment_plan",
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "status": "PASS",
        "readiness": "DETERMINISTIC_PLAN_ONLY",
        "context": dict(context),
        "config_sha256": plan["config_sha256"].lower(),
        "dataset_sha256": plan["dataset_sha256"].lower(),
        "code_commit_sha": commit.lower(),
        "expected_seeds": seeds,
        "expected_folds": [str(fold) for fold in folds],
        "expected_windows": normalized_windows,
        "expected_run_ids": expected_run_ids,
        "expected_runs": run_count,
        "coverage": {
            "seed_count": len(seeds),
            "fold_count": len(folds),
            "window_count": len(normalized_windows),
            "cartesian_coverage_complete": len(run_matrix) == run_count,
        },
        "run_matrix": run_matrix,
        "safety_flags": dict(FALSE_FLAGS),
        **FALSE_FLAGS,
        "jobs_launched": False,
        "market_data_read": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "scope_limitations": [
            "This output is a deterministic plan, not evidence that any experiment ran.",
            "The planner does not inspect raw market data or verify the supplied dataset/configuration hashes against local files.",
            "A complete matrix does not establish predictive validity, profitability, or deployment readiness.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON declarative experiment plan")
    parser.add_argument("--output", type=Path, help="New plan output path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        plan = strict_json_loads(args.input.read_text(encoding="utf-8"))
        report = build_experiment_plan(plan)
        output = args.output or args.input.with_name(args.input.stem + "_plan.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "plan_id": report["plan_id"],
            "plan_sha256": report["plan_sha256"],
            "expected_runs": report["expected_runs"],
            "plan_path": str(output.resolve()),
            "jobs_launched": False,
        }, indent=2))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit evaluation-row lineage against declared windows for offline Forex research.

The audit reads evaluation-row CSV files referenced by each run manifest and verifies every
timestamp falls within the planned half-open interval [start, end). It checks dataset,
evaluation-code, row-file, and metrics-artifact digests plus row counts. It does not rerun
metric calculations, execute experiments, train/select models, or connect to trading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
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
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _confined_file(root: Path, relative_value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(relative_value, str) or not relative_value.strip():
        return None, "path_invalid"
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, "path_must_be_confined"
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, "path_escapes_root"
    return resolved, None


def audit_evaluation_window_lineage(
    plan: dict[str, Any],
    registry_report: dict[str, Any],
    *,
    registry_root: Path,
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("audit_name") != "research_experiment_plan":
        raise ValueError("plan_report_invalid")
    if plan.get("status") != "PASS":
        raise ValueError("plan_report_not_passing")
    if not isinstance(registry_report, dict) or registry_report.get("audit_name") != "research_experiment_registry":
        raise ValueError("registry_report_invalid")
    if registry_report.get("status") not in PASS_STATUSES:
        raise ValueError("registry_report_not_passing")

    matrix = plan.get("run_matrix")
    records = registry_report.get("run_results")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("plan_run_matrix_invalid")
    if not isinstance(records, list):
        raise ValueError("registry_run_results_invalid")

    planned: dict[str, dict[str, Any]] = {}
    for item in matrix:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str) or not item["run_id"]:
            raise ValueError("plan_run_identity_invalid")
        if item["run_id"] in planned:
            raise ValueError("plan_run_id_duplicate")
        planned[item["run_id"]] = item

    root = registry_root.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    runs_checked = 0
    rows_checked = 0
    row_files_checked = 0
    metric_artifacts_checked = 0
    metric_artifact_bytes = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append({"code": "registry_record_invalid", "index": index})
            continue
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            errors.append({"code": "registry_run_id_invalid", "index": index})
            continue
        if run_id in seen:
            errors.append({"code": "duplicate_registry_run_id", "run_id": run_id})
            continue
        seen.add(run_id)
        expected = planned.get(run_id)
        if expected is None:
            warnings.append(f"unplanned_run_skipped:{run_id}")
            continue
        if record.get("status") != "PASS":
            errors.append({"code": "registry_run_not_passed", "run_id": run_id})
            continue

        manifest_path, path_error = _confined_file(root, record.get("manifest_path"))
        if path_error or manifest_path is None:
            errors.append({"code": f"manifest_{path_error or 'path_invalid'}", "run_id": run_id})
            continue
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except OSError:
            errors.append({"code": "manifest_unreadable", "run_id": run_id})
            continue
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append({"code": "manifest_json_invalid", "run_id": run_id})
            continue
        if not isinstance(manifest, dict):
            errors.append({"code": "manifest_root_invalid", "run_id": run_id})
            continue
        if manifest.get("run_id") != run_id:
            errors.append({"code": "manifest_run_id_mismatch", "run_id": run_id})
        if record.get("manifest_sha256") != _sha256(manifest_raw):
            errors.append({"code": "manifest_sha256_mismatch", "run_id": run_id})

        evidence = manifest.get("evaluation_evidence")
        if not isinstance(evidence, dict):
            errors.append({"code": "evaluation_evidence_missing", "run_id": run_id})
            continue
        if evidence.get("dataset_sha256") != expected.get("dataset_sha256"):
            errors.append({"code": "evaluation_dataset_digest_mismatch", "run_id": run_id})
        code_digest = evidence.get("evaluation_code_sha256")
        if not _valid_digest(code_digest):
            errors.append({"code": "evaluation_code_digest_invalid", "run_id": run_id})
        declared_window = expected.get("evaluation_window")
        if not isinstance(declared_window, dict):
            errors.append({"code": "planned_evaluation_window_invalid", "run_id": run_id})
            continue
        start = _timestamp(declared_window.get("start"))
        end = _timestamp(declared_window.get("end"))
        if start is None or end is None or start >= end:
            errors.append({"code": "planned_evaluation_window_invalid", "run_id": run_id})
            continue

        rows_path, rows_path_error = _confined_file(manifest_path.parent, evidence.get("evaluation_rows_path"))
        if rows_path_error or rows_path is None:
            errors.append({"code": f"evaluation_rows_{rows_path_error or 'path_invalid'}", "run_id": run_id})
            continue
        try:
            rows_raw = rows_path.read_bytes()
        except OSError:
            errors.append({"code": "evaluation_rows_unreadable", "run_id": run_id})
            continue
        row_digest = evidence.get("evaluation_rows_sha256")
        if not _valid_digest(row_digest) or _sha256(rows_raw) != row_digest.lower():
            errors.append({"code": "evaluation_rows_sha256_mismatch_or_invalid", "run_id": run_id})
            continue

        timestamp_column = evidence.get("timestamp_column", "timestamp")
        if not isinstance(timestamp_column, str) or not timestamp_column.strip():
            errors.append({"code": "timestamp_column_invalid", "run_id": run_id})
            continue
        try:
            text = rows_raw.decode("utf-8-sig")
            reader = csv.DictReader(text.splitlines())
            if not reader.fieldnames or timestamp_column not in reader.fieldnames:
                errors.append({"code": "evaluation_rows_timestamp_column_missing", "run_id": run_id})
                continue
            row_count = 0
            observed_times: list[datetime] = []
            for row_index, row in enumerate(reader, start=2):
                parsed = _timestamp(row.get(timestamp_column))
                if parsed is None:
                    errors.append({"code": "evaluation_row_timestamp_invalid", "run_id": run_id, "row": row_index})
                    continue
                if parsed.utcoffset().total_seconds() != 0:
                    errors.append({"code": "evaluation_row_timestamp_not_utc", "run_id": run_id, "row": row_index})
                    continue
                if not (start <= parsed < end):
                    errors.append({
                        "code": "evaluation_row_outside_declared_window",
                        "run_id": run_id,
                        "row": row_index,
                        "timestamp": row.get(timestamp_column),
                        "window_start": declared_window.get("start"),
                        "window_end": declared_window.get("end"),
                    })
                row_count += 1
                observed_times.append(parsed)
        except (UnicodeDecodeError, csv.Error) as exc:
            errors.append({"code": "evaluation_rows_csv_invalid", "run_id": run_id, "detail": str(exc)})
            continue

        declared_count = evidence.get("evaluated_row_count")
        if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count < 1:
            errors.append({"code": "evaluated_row_count_invalid", "run_id": run_id})
        elif declared_count != row_count:
            errors.append({
                "code": "evaluated_row_count_mismatch",
                "run_id": run_id,
                "declared": declared_count,
                "observed": row_count,
            })
        if row_count == 0:
            errors.append({"code": "evaluation_rows_empty", "run_id": run_id})
        if observed_times:
            first_value = evidence.get("first_evaluated_timestamp")
            last_value = evidence.get("last_evaluated_timestamp")
            first = _timestamp(first_value)
            last = _timestamp(last_value)
            if first is None or first != min(observed_times):
                errors.append({"code": "first_evaluated_timestamp_mismatch", "run_id": run_id})
            if last is None or last != max(observed_times):
                errors.append({"code": "last_evaluated_timestamp_mismatch", "run_id": run_id})

        metric_path_value = evidence.get("metrics_artifact_path")
        metric_path, metric_path_error = _confined_file(manifest_path.parent, metric_path_value)
        if metric_path_error or metric_path is None:
            errors.append({"code": f"metrics_artifact_{metric_path_error or 'path_invalid'}", "run_id": run_id})
        else:
            try:
                metric_raw = metric_path.read_bytes()
            except OSError:
                errors.append({"code": "metrics_artifact_unreadable", "run_id": run_id})
            else:
                metric_digest = evidence.get("metrics_artifact_sha256")
                if not _valid_digest(metric_digest) or _sha256(metric_raw) != metric_digest.lower():
                    errors.append({"code": "metrics_artifact_sha256_mismatch_or_invalid", "run_id": run_id})
                elif not metric_raw:
                    errors.append({"code": "metrics_artifact_empty", "run_id": run_id})
                else:
                    metric_artifacts_checked += 1
                    metric_artifact_bytes += len(metric_raw)
        rows_checked += row_count
        row_files_checked += 1
        runs_checked += 1

    for run_id in sorted(set(planned) - seen):
        errors.append({"code": "planned_run_missing_from_registry", "run_id": run_id})

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_evaluation_window_lineage",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "registry_id": registry_report.get("registry_id"),
        "status": status,
        "readiness": "NOT_WINDOW_LINEAGE_VERIFIED" if errors else (
            "WINDOW_LINEAGE_VERIFIED_WITH_WARNINGS" if warnings else "EVALUATION_ROW_WINDOW_LINEAGE_VERIFIED"
        ),
        "planned_runs": len(planned),
        "runs_checked": runs_checked,
        "evaluation_row_files_checked": row_files_checked,
        "evaluation_rows_checked": rows_checked,
        "metrics_artifacts_checked": metric_artifacts_checked,
        "metrics_artifact_bytes_checked": metric_artifact_bytes,
        "errors": errors,
        "warnings": warnings,
        **FALSE_FLAGS,
        "deployment_authorized": False,
        "scope_limitations": [
            "Checks each row timestamp in the declared evaluation CSV against the planned half-open UTC interval [start, end).",
            "Checks declared dataset/evaluation-code digests and file hashes; it does not prove the evaluation code actually generated the metrics.",
            "Does not recompute metric values, execute experiments, train/select models, or connect to broker/trading systems.",
            "Passing is data-lineage evidence only and does not establish strategy quality, profitability, or deployment suitability.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registry-report", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_evaluation_window_lineage(
            _read_object(args.plan),
            _read_object(args.registry_report),
            registry_root=args.registry_root,
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "runs_checked": report["runs_checked"],
            "evaluation_rows_checked": report["evaluation_rows_checked"],
            "metrics_artifacts_checked": report["metrics_artifacts_checked"],
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

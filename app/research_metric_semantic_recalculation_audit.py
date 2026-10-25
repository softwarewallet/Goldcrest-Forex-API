"""Independently recalculate explicitly declared simple metrics from offline evaluation rows.

This audit only supports row_count, sum, mean, min, max, and median over named numeric CSV
columns. It does not infer strategy PnL semantics or authorize trading/deployment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _confined(root: Path, value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "path_invalid"
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, "path_must_be_confined"
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError:
        return None, "path_escapes_root"
    return result, None


def _read_json(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _calculate(kind: str, values: list[float]) -> float | int | None:
    if kind == "row_count":
        return len(values)
    if not values:
        return None
    if kind == "sum":
        return math.fsum(values)
    if kind == "mean":
        return statistics.fmean(values)
    if kind == "min":
        return min(values)
    if kind == "max":
        return max(values)
    if kind == "median":
        return statistics.median(values)
    raise ValueError("unsupported_metric_kind")


def audit_metric_semantics(
    plan: dict[str, Any],
    registry_report: dict[str, Any],
    *,
    registry_root: Path,
    abs_tolerance: float = 1e-9,
    rel_tolerance: float = 1e-7,
) -> dict[str, Any]:
    if plan.get("audit_name") != "research_experiment_plan" or plan.get("status") != "PASS":
        raise ValueError("plan_report_invalid_or_not_passing")
    if registry_report.get("audit_name") != "research_experiment_registry" or registry_report.get("status") not in PASS_STATUSES:
        raise ValueError("registry_report_invalid_or_not_passing")
    if not math.isfinite(abs_tolerance) or abs_tolerance < 0 or not math.isfinite(rel_tolerance) or rel_tolerance < 0:
        raise ValueError("tolerance_invalid")
    matrix, records = plan.get("run_matrix"), registry_report.get("run_results")
    if not isinstance(matrix, list) or not matrix or not isinstance(records, list):
        raise ValueError("plan_or_registry_rows_invalid")
    planned: dict[str, dict[str, Any]] = {}
    for item in matrix:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str) or not item["run_id"] or item["run_id"] in planned:
            raise ValueError("plan_run_identity_invalid_or_duplicate")
        planned[item["run_id"]] = item

    root = registry_root.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    runs_checked = metrics_checked = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str) or not record["run_id"]:
            errors.append({"code": "registry_record_identity_invalid", "index": index})
            continue
        run_id = record["run_id"]
        if run_id in seen:
            errors.append({"code": "duplicate_registry_run_id", "run_id": run_id})
            continue
        seen.add(run_id)
        if run_id not in planned:
            warnings.append(f"unplanned_run_skipped:{run_id}")
            continue
        if record.get("status") != "PASS":
            errors.append({"code": "registry_run_not_passed", "run_id": run_id})
            continue
        manifest_path, path_error = _confined(root, record.get("manifest_path"))
        if path_error or manifest_path is None:
            errors.append({"code": f"manifest_{path_error or 'path_invalid'}", "run_id": run_id})
            continue
        try:
            manifest_raw = manifest_path.read_bytes()
        except OSError:
            errors.append({"code": "manifest_unreadable", "run_id": run_id})
            continue
        manifest = _read_json(manifest_raw)
        if manifest is None:
            errors.append({"code": "manifest_json_invalid", "run_id": run_id})
            continue
        if manifest.get("run_id") != run_id:
            errors.append({"code": "manifest_run_id_mismatch", "run_id": run_id})
        if record.get("manifest_sha256") != _sha(manifest_raw):
            errors.append({"code": "manifest_sha256_mismatch", "run_id": run_id})
        evidence = manifest.get("evaluation_evidence")
        if not isinstance(evidence, dict):
            errors.append({"code": "evaluation_evidence_missing", "run_id": run_id})
            continue
        contract = evidence.get("metric_recalculation_contract")
        if not isinstance(contract, dict) or contract.get("version") != "1" or not isinstance(contract.get("metrics"), dict) or not contract["metrics"]:
            errors.append({"code": "metric_recalculation_contract_missing_or_invalid", "run_id": run_id})
            continue
        if not _digest(evidence.get("evaluation_code_sha256")):
            errors.append({"code": "evaluation_code_digest_invalid", "run_id": run_id})
        if evidence.get("dataset_sha256") != planned[run_id].get("dataset_sha256"):
            errors.append({"code": "evaluation_dataset_digest_mismatch", "run_id": run_id})

        rows_path, rows_error = _confined(manifest_path.parent, evidence.get("evaluation_rows_path"))
        metric_path, metric_error = _confined(manifest_path.parent, evidence.get("metrics_artifact_path"))
        if rows_error or rows_path is None:
            errors.append({"code": f"evaluation_rows_{rows_error or 'path_invalid'}", "run_id": run_id})
            continue
        if metric_error or metric_path is None:
            errors.append({"code": f"metrics_artifact_{metric_error or 'path_invalid'}", "run_id": run_id})
            continue
        try:
            rows_raw, metric_raw = rows_path.read_bytes(), metric_path.read_bytes()
        except OSError:
            errors.append({"code": "input_or_metrics_artifact_unreadable", "run_id": run_id})
            continue
        if not _digest(evidence.get("evaluation_rows_sha256")) or _sha(rows_raw) != evidence["evaluation_rows_sha256"].lower():
            errors.append({"code": "evaluation_rows_sha256_mismatch_or_invalid", "run_id": run_id})
            continue
        if not _digest(evidence.get("metrics_artifact_sha256")) or _sha(metric_raw) != evidence["metrics_artifact_sha256"].lower():
            errors.append({"code": "metrics_artifact_sha256_mismatch_or_invalid", "run_id": run_id})
            continue
        artifact = _read_json(metric_raw)
        if artifact is None or not isinstance(artifact.get("metrics"), dict):
            errors.append({"code": "metrics_artifact_schema_invalid", "run_id": run_id})
            continue
        try:
            reader = csv.DictReader(rows_raw.decode("utf-8-sig").splitlines())
            if not reader.fieldnames:
                raise ValueError("evaluation_rows_header_missing")
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error, ValueError) as exc:
            errors.append({"code": "evaluation_rows_csv_invalid", "run_id": run_id, "detail": str(exc)})
            continue

        run_has_error = False
        for metric_name, spec in contract["metrics"].items():
            if not isinstance(metric_name, str) or not metric_name or not isinstance(spec, dict):
                errors.append({"code": "metric_contract_entry_invalid", "run_id": run_id})
                run_has_error = True
                continue
            kind = spec.get("kind")
            if kind == "row_count":
                column_values = [1.0] * len(rows)
            else:
                column = spec.get("column")
                if kind not in {"sum", "mean", "min", "max", "median"} or not isinstance(column, str) or not column or column not in (reader.fieldnames or []):
                    errors.append({"code": "metric_contract_kind_or_column_invalid", "run_id": run_id, "metric": metric_name})
                    run_has_error = True
                    continue
                column_values = []
                invalid_row = False
                for row_number, row in enumerate(rows, start=2):
                    try:
                        value = float(row.get(column, ""))
                    except (TypeError, ValueError):
                        value = math.nan
                    if not math.isfinite(value):
                        errors.append({"code": "metric_input_value_invalid", "run_id": run_id, "metric": metric_name, "row": row_number})
                        invalid_row = True
                        break
                    column_values.append(value)
                if invalid_row:
                    run_has_error = True
                    continue
            try:
                calculated = _calculate(kind, column_values)
            except ValueError:
                errors.append({"code": "metric_contract_kind_invalid", "run_id": run_id, "metric": metric_name})
                run_has_error = True
                continue
            declared = artifact["metrics"].get(metric_name)
            if calculated is None or isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(float(declared)):
                errors.append({"code": "metric_value_missing_or_nonfinite", "run_id": run_id, "metric": metric_name})
                run_has_error = True
                continue
            if not math.isclose(float(calculated), float(declared), rel_tol=rel_tolerance, abs_tol=abs_tolerance):
                errors.append({"code": "metric_recalculation_mismatch", "run_id": run_id, "metric": metric_name, "calculated": calculated, "declared": declared})
                run_has_error = True
            else:
                metrics_checked += 1
        if not run_has_error:
            runs_checked += 1

    for run_id in sorted(set(planned) - seen):
        errors.append({"code": "planned_run_missing_from_registry", "run_id": run_id})
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_metric_semantic_recalculation",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "registry_id": registry_report.get("registry_id"),
        "status": status,
        "readiness": "NOT_METRIC_SEMANTICS_VERIFIED" if errors else ("METRIC_SEMANTICS_VERIFIED_WITH_WARNINGS" if warnings else "DECLARED_METRICS_INDEPENDENTLY_RECALCULATED"),
        "planned_runs": len(planned),
        "runs_checked": runs_checked,
        "metrics_recalculated": metrics_checked,
        "errors": errors,
        "warnings": warnings,
        **FALSE_FLAGS,
        "deployment_authorized": False,
        "scope_limitations": [
            "Independently recalculates only explicitly contracted row_count, sum, mean, min, max, and median metrics from named numeric CSV columns.",
            "Does not infer net PnL, drawdown, trade costs, position sizing, or strategy semantics; those require an explicit, separately validated calculation contract.",
            "Hashes establish byte integrity, not proof that the declared evaluation code generated the artifact.",
            "Offline report-only audit; does not execute experiments, train/select models, connect to brokers, or authorize deployment.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registry-report", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_metric_semantics(_read_object(args.plan), _read_object(args.registry_report), registry_root=args.registry_root)
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "readiness": report["readiness"], "runs_checked": report["runs_checked"], "metrics_recalculated": report["metrics_recalculated"], "error_count": len(report["errors"]), "report_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

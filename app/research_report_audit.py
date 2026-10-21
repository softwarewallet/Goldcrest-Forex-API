"""Integrity and consistency audit for offline research pipeline artifacts.

This module verifies hashes and research-only invariants; it does not run models,
fetch data, choose a winner, or connect to broker/execution systems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _resolve_artifact(raw: str, manifest_path: Path, base_dir: Path | None) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    roots = [base_dir] if base_dir is not None else []
    roots.extend([Path.cwd(), manifest_path.parent])
    for root in roots:
        resolved = root / candidate
        if resolved.is_file():
            return resolved
    # Return the most intuitive path so the missing-artifact check is useful.
    return (base_dir or Path.cwd()) / candidate


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _verify_embedded_hash(payload: dict[str, Any], field: str) -> bool:
    actual = payload.get(field)
    if not isinstance(actual, str) or len(actual) != 64:
        return False
    body = dict(payload)
    body.pop(field, None)
    return _sha256_bytes(_canonical(body)) == actual


def _metric_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    by_metric: dict[str, list[float]] = {}
    for experiment in matrix.get("experiments", []):
        for fold in experiment.get("fold_results", []):
            metrics = fold.get("test_metrics", {})
            for name in ("brier_score", "log_loss", "accuracy_at_fixed_half", "expected_calibration_error"):
                value = metrics.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    by_metric.setdefault(name, []).append(float(value))
    summary: dict[str, Any] = {}
    for name, values in sorted(by_metric.items()):
        summary[name] = {
            "fold_observations": len(values),
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def audit_research_pipeline(manifest_path: str | Path, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Audit pipeline, dataset and matrix manifests plus referenced file hashes."""
    manifest_file = Path(manifest_path).resolve()
    base = Path(base_dir).resolve() if base_dir is not None else None
    manifest = _read_json(manifest_file)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    _check(checks, "pipeline_manifest_hash", _verify_embedded_hash(manifest, "pipeline_manifest_sha256"),
           "Canonical pipeline manifest hash matches." if _verify_embedded_hash(manifest, "pipeline_manifest_sha256") else "Pipeline manifest hash is missing or mismatched.")
    _check(checks, "research_only_flags",
           manifest.get("actionable") is False and manifest.get("execution_connected") is False
           and manifest.get("automatic_winner_selection") is False
           and manifest.get("test_metrics_used_for_selection") is False,
           "Pipeline explicitly remains non-actionable, disconnected from execution, and does not select on test metrics.")

    required_paths = {
        "raw_ohlc_path": "raw_ohlc_sha256",
        "dataset_path": "dataset_sha256",
        "dataset_manifest_path": "dataset_manifest_sha256",
        "experiment_matrix_report": "experiment_matrix_report_sha256",
    }
    resolved: dict[str, Path] = {}
    for path_field, hash_field in required_paths.items():
        raw_path = manifest.get(path_field)
        expected_hash = manifest.get(hash_field)
        if not isinstance(raw_path, str) or not raw_path:
            _check(checks, f"{path_field}_exists", False, f"Missing path field {path_field}.")
            continue
        path = _resolve_artifact(raw_path, manifest_file, base)
        resolved[path_field] = path
        exists = path.is_file()
        _check(checks, f"{path_field}_exists", exists, f"Artifact exists: {path}" if exists else f"Artifact not found: {path}")
        if exists:
            actual_hash = _sha256_file(path)
            _check(checks, f"{path_field}_sha256", isinstance(expected_hash, str) and actual_hash == expected_hash,
                   "Referenced file hash matches pipeline manifest." if actual_hash == expected_hash else "Referenced file hash differs from pipeline manifest.")
        else:
            _check(checks, f"{path_field}_sha256", False, "Cannot verify hash because artifact is missing.")

    dataset_manifest: dict[str, Any] | None = None
    if "dataset_manifest_path" in resolved and resolved["dataset_manifest_path"].is_file():
        try:
            dataset_manifest = _read_json(resolved["dataset_manifest_path"])
            valid = _verify_embedded_hash(dataset_manifest, "manifest_sha256")
            _check(checks, "dataset_manifest_internal_hash", valid,
                   "Dataset manifest's internal hash matches." if valid else "Dataset manifest internal hash is missing or mismatched.")
            _check(checks, "dataset_manifest_non_actionable", dataset_manifest.get("actionable") is False,
                   "Dataset manifest is marked non-actionable.")
            _check(checks, "dataset_snapshot_hash_consistency",
                   dataset_manifest.get("output_dataset_sha256") == manifest.get("dataset_sha256"),
                   "Dataset manifest output hash agrees with pipeline manifest.")
        except ValueError as exc:
            _check(checks, "dataset_manifest_parse", False, str(exc))

    matrix: dict[str, Any] | None = None
    if "experiment_matrix_report" in resolved and resolved["experiment_matrix_report"].is_file():
        try:
            matrix = _read_json(resolved["experiment_matrix_report"])
            valid = _verify_embedded_hash(matrix, "report_payload_sha256")
            _check(checks, "matrix_internal_hash", valid,
                   "Experiment matrix report's internal hash matches." if valid else "Experiment matrix report internal hash is missing or mismatched.")
            _check(checks, "matrix_research_only",
                   matrix.get("actionable") is False
                   and matrix.get("automatic_winner_selection") is False
                   and matrix.get("test_metrics_used_for_selection") is False,
                   "Experiment matrix is marked non-actionable with no automatic winner selection.")
            _check(checks, "matrix_dataset_hash_consistency",
                   matrix.get("dataset_sha256") == manifest.get("dataset_sha256"),
                   "Experiment matrix references the same dataset hash.")
            experiments = matrix.get("experiments")
            valid_experiments = isinstance(experiments, list) and bool(experiments)
            _check(checks, "matrix_has_experiments", valid_experiments,
                   f"Experiment count: {len(experiments) if isinstance(experiments, list) else 0}.")
            if valid_experiments:
                windows = matrix.get("common_fold_test_windows")
                windows_ok = isinstance(windows, list) and bool(windows)
                fold_counts_ok = True
                test_flags_ok = True
                expected_windows = {
                    (w.get("fold_index"), w.get("test_start_utc"), w.get("test_end_utc"), w.get("test_rows"))
                    for w in windows if isinstance(w, dict)
                } if isinstance(windows, list) else set()
                for experiment in experiments:
                    folds = experiment.get("fold_results", [])
                    fold_counts_ok = fold_counts_ok and len(folds) == matrix.get("fold_count")
                    actual_windows = {
                        (f.get("fold_index"), f.get("test_start_utc"), f.get("test_end_utc"),
                         f.get("test_metrics", {}).get("sample_count"))
                        for f in folds if isinstance(f, dict)
                    }
                    windows_ok = windows_ok and actual_windows == expected_windows
                    test_flags_ok = test_flags_ok and all(
                        f.get("test_metrics_used_for_tuning") is False for f in folds
                    )
                _check(checks, "matrix_fold_counts", fold_counts_ok, "Each experiment has the declared number of folds." if fold_counts_ok else "Fold counts do not match the matrix declaration.")
                _check(checks, "common_test_windows", windows_ok, "Experiments use the declared common test windows." if windows_ok else "One or more experiment folds do not match common test windows.")
                _check(checks, "test_metrics_not_used_for_tuning", test_flags_ok, "Fold records state test metrics were not used for tuning." if test_flags_ok else "A fold is missing the expected test-isolation flag.")
        except (ValueError, TypeError, AttributeError) as exc:
            _check(checks, "matrix_parse_and_consistency", False, f"Could not fully audit matrix: {exc}")

    summaries = _metric_summary(matrix) if matrix is not None else {}
    if not summaries:
        warnings.append("No fold-level predictive metrics were available to summarize.")
    warnings.append("Predictive metrics are not profitability evidence; this audit does not authorize deployment or trading.")
    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_ARTIFACT_AUDIT",
        "manifest_path": str(manifest_file),
        "audit_passed": not failed,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "fold_metric_summary": summaries,
        "warnings": warnings,
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Goldcrest offline research artifacts and hash consistency.")
    parser.add_argument("--pipeline-manifest", required=True, help="Path to *_pipeline_manifest.json")
    parser.add_argument("--base-dir", help="Optional root for relative artifact paths")
    parser.add_argument("--output", help="Optional path for the JSON audit report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_research_pipeline(args.pipeline_manifest, base_dir=args.base_dir)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0 if report["audit_passed"] else 1
    except Exception as exc:
        print(f"research report audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

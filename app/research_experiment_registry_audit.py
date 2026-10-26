"""Offline registry audit for reproducible experiment manifests and artifact lineage.

The registry verifies manifest digests and delegates manifest/artifact checks to the
provenance validator. It summarizes run coverage and descriptive metric dispersion only;
it does not run experiments, train/select models, infer significance, or touch execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

from app.research_experiment_provenance import validate_experiment_manifest, valid_sha256


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit_experiment_registry(registry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise ValueError("registry_must_be_object")
    if registry.get("schema_version") != 1:
        raise ValueError("registry_schema_version_unsupported")
    registry_id = registry.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id.strip():
        raise ValueError("registry_id_invalid")
    entries = registry.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("runs_must_be_nonempty_list")

    root = base_dir.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    run_results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_run_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"code": "registry_entry_invalid", "index": index})
            continue
        rel = entry.get("manifest_path")
        if not isinstance(rel, str) or not rel:
            errors.append({"code": "manifest_path_invalid", "index": index})
            continue
        if rel in seen_paths:
            errors.append({"code": "manifest_path_duplicate", "path": rel})
            continue
        seen_paths.add(rel)
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            errors.append({"code": "manifest_path_must_be_confined", "path": rel})
            continue
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append({"code": "manifest_path_escapes_registry_root", "path": rel})
            continue
        try:
            raw = resolved.read_bytes()
            manifest = _load_json_object(raw, source=str(resolved))
        except OSError:
            errors.append({"code": "manifest_unreadable", "path": rel})
            continue
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            errors.append({"code": "manifest_json_invalid", "path": rel})
            continue
        if not isinstance(manifest, dict):
            errors.append({"code": "manifest_root_must_be_object", "path": rel})
            continue
        actual_hash = _sha256(raw)
        expected_hash = entry.get("expected_manifest_sha256")
        item_errors: list[str] = []
        if not valid_sha256(expected_hash):
            item_errors.append("expected_manifest_sha256_invalid")
        elif actual_hash != expected_hash.lower():
            item_errors.append("manifest_sha256_mismatch")
        report = validate_experiment_manifest(manifest, base_dir=resolved.parent)
        if report["status"] != "PASS":
            item_errors.append("manifest_provenance_validation_failed")
        run_id = report.get("run_id")
        if isinstance(run_id, str) and run_id:
            if run_id in seen_run_ids:
                item_errors.append("duplicate_run_id")
            seen_run_ids.add(run_id)
        for code in item_errors:
            errors.append({"code": code, "index": index, "path": rel, "run_id": run_id})
        run_results.append({
            "manifest_path": rel,
            "manifest_sha256": actual_hash,
            "run_id": run_id,
            "seed": report.get("seed"),
            "fold_id": report.get("fold_id"),
            "window_id": manifest.get("window_id"),
            "evaluation_window": manifest.get("evaluation_window"),
            "context": report.get("context"),
            "config_sha256": manifest.get("config_sha256"),
            "dataset_sha256": report.get("dataset_sha256"),
            "code_commit_sha": report.get("code_commit_sha"),
            "status": "PASS" if not item_errors else "FAIL",
            "errors": item_errors,
            "provenance_status": report["status"],
        })
        if not item_errors:
            validated.append(report)

    expected_ids = registry.get("expected_run_ids")
    if expected_ids is not None:
        if (not isinstance(expected_ids, list) or
                any(not isinstance(item, str) or not item.strip() for item in expected_ids) or
                len(set(expected_ids)) != len(expected_ids)):
            errors.append({"code": "expected_run_ids_invalid"})
        else:
            actual_ids = {r.get("run_id") for r in validated}
            for missing in sorted(set(expected_ids) - actual_ids):
                errors.append({"code": "expected_run_missing", "run_id": missing})
            for unexpected in sorted(actual_ids - set(expected_ids)):
                warnings.append(f"unexpected_run_present:{unexpected}")

    expected_seeds = registry.get("expected_seeds")
    expected_folds = registry.get("expected_folds")
    if expected_seeds is not None:
        if (not isinstance(expected_seeds, list) or not expected_seeds or
                any(not isinstance(x, int) or isinstance(x, bool) for x in expected_seeds) or
                len(set(expected_seeds)) != len(expected_seeds)):
            errors.append({"code": "expected_seeds_invalid"})
        else:
            actual = {r.get("seed") for r in validated}
            for seed in sorted(set(expected_seeds) - actual):
                errors.append({"code": "expected_seed_missing", "seed": seed})
    if expected_folds is not None:
        if (not isinstance(expected_folds, list) or not expected_folds or
                any(not isinstance(x, (str, int)) or isinstance(x, bool) for x in expected_folds) or
                len({str(x) for x in expected_folds}) != len(expected_folds)):
            errors.append({"code": "expected_folds_invalid"})
        else:
            expected_fold_set = {str(x) for x in expected_folds}
            for seed in sorted({r.get("seed") for r in validated if isinstance(r.get("seed"), int)}):
                actual = {r.get("fold_id") for r in validated if r.get("seed") == seed}
                for fold in sorted(expected_fold_set - actual):
                    errors.append({"code": "seed_fold_coverage_incomplete", "seed": seed, "fold_id": fold})

    if validated:
        for field in ("context", "dataset_sha256"):
            values = {json.dumps(r.get(field), sort_keys=True) for r in validated}
            if len(values) > 1:
                errors.append({"code": f"registry_{field}_mismatch"})
        metric_sets = [set(r.get("metrics", {})) for r in validated]
        if any(m != metric_sets[0] for m in metric_sets[1:]):
            errors.append({"code": "registry_metric_set_mismatch"})
    metric_summary: dict[str, Any] = {}
    metric_names = sorted(set.intersection(*(set(r.get("metrics", {})) for r in validated))) if validated else []
    for name in metric_names:
        values = [r["metrics"][name] for r in validated]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in values):
            errors.append({"code": f"registry_metric_invalid:{name}"})
            continue
        metric_summary[name] = {
            "observations": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "population_standard_deviation": statistics.pstdev(values),
        }

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_experiment_registry",
        "registry_id": registry_id,
        "status": status,
        "readiness": "NOT_READY" if errors else "DESCRIPTIVE_RESEARCH_REGISTRY_ONLY",
        "runs_supplied": len(entries),
        "runs_validated": len(validated),
        "run_results": run_results,
        "metric_summaries": metric_summary,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry = _load_json_object(registry_path.read_bytes(), source=str(registry_path))
    report = audit_experiment_registry(registry, base_dir=registry_path.parent)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-run protocol and artifact registry for offline research experiments.

The registry audits supplied persisted manifests; it never runs experiments, selects models,
recomputes metrics, connects to execution, or authorizes deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.research_run_audit import audit_research_run_manifest
from app.research_strict_json import strict_json_loads

_PROTOCOL_FIELDS = (
    "symbol",
    "timeframe",
    "source",
    "source_version",
    "terms_reference_sha256",
    "raw_ohlc_sha256",
    "dataset_sha256",
    "feature_schema_sha256",
    "feature_implementation_sha256",
    "label_horizon_bars",
    "class_1_means",
    "max_label_elapsed_hours",
    "label_eligibility_mode",
)
_CONFIG_PROTOCOL_FIELDS = (
    "initial_train_bars",
    "validation_bars",
    "test_bars",
    "step_bars",
    "trials_per_fold",
    "bootstrap_resamples",
    "max_experiments",
    "include_ablation",
)
_REQUIRED_PASS_STATUSES = (
    "raw_ohlc_audit_status",
    "feature_source_audit_status",
    "point_in_time_feature_invariance_audit_status",
    "label_maturity_audit_status",
    "provenance_audit_status",
    "temporal_calibration_audit_status",
    "model_evaluation_gate_status",
)
_FALSE_FLAGS = (
    "actionable",
    "execution_connected",
    "automatic_winner_selection",
    "test_metrics_used_for_selection",
)


def _read_manifest(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest_root_must_be_object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_experiment_registry(manifest_paths: list[str | Path]) -> dict[str, Any]:
    """Validate all runs and require a consistent protocol with distinct seeds."""
    if len(manifest_paths) < 2:
        raise ValueError("at_least_two_manifests_required_for_cross_run_qualification")

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    run_results: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_seeds: dict[int, str] = {}
    protocol_values: dict[str, set[str]] = {field: set() for field in _PROTOCOL_FIELDS}
    protocol_values.update({f"research_config.{field}": set() for field in _CONFIG_PROTOCOL_FIELDS})
    dataset_fingerprints: dict[str, list[str]] = {}

    for index, raw_path in enumerate(manifest_paths):
        path = Path(raw_path).expanduser().resolve()
        run_errors: list[str] = []
        if path in seen_paths:
            run_errors.append("manifest_path_duplicate")
        seen_paths.add(path)

        try:
            manifest = _read_manifest(path)
            manifest_hash = _sha256(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            manifest = {}
            manifest_hash = None
            run_errors.append(f"manifest_unreadable:{type(exc).__name__}")

        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            run_errors.append("run_id_missing")
            run_id = f"invalid_run_{index}"
        elif run_id in seen_run_ids:
            run_errors.append("run_id_duplicate")
        seen_run_ids.add(run_id)

        integrity = audit_research_run_manifest(path) if manifest else {
            "status": "FAIL", "errors": ["manifest_unreadable"], "warnings": []
        }
        if integrity.get("status") != "PASS":
            run_errors.append(f"cross_artifact_integrity_not_clean_pass:{integrity.get('status')}")
        if integrity.get("errors"):
            run_errors.append("cross_artifact_integrity_errors_present")
        if integrity.get("warnings"):
            warnings.append(f"{run_id}:cross_artifact_warnings_present")

        for field in _FALSE_FLAGS:
            if manifest.get(field) is not False:
                run_errors.append(f"safety_flag_not_false:{field}")

        for field in _REQUIRED_PASS_STATUSES:
            if manifest.get(field) != "PASS":
                run_errors.append(f"required_status_not_clean_pass:{field}")

        for field in _PROTOCOL_FIELDS:
            value = manifest.get(field)
            if value is None:
                run_errors.append(f"protocol_field_missing:{field}")
            else:
                protocol_values[field].add(json.dumps(value, sort_keys=True, allow_nan=False))

        config = manifest.get("research_config")
        if not isinstance(config, dict):
            run_errors.append("research_config_missing")
            seed = None
        else:
            seed = config.get("random_seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                run_errors.append("random_seed_invalid")
                seed = None
            elif seed in seen_seeds:
                run_errors.append(f"random_seed_reused:{seen_seeds[seed]}")
            else:
                seen_seeds[seed] = run_id

            for field in _CONFIG_PROTOCOL_FIELDS:
                value = config.get(field)
                key = f"research_config.{field}"
                if value is None:
                    run_errors.append(f"research_config_field_missing:{field}")
                else:
                    protocol_values.setdefault(key, set()).add(
                        json.dumps(value, sort_keys=True, allow_nan=False)
                    )

        reproducibility_path = manifest.get("reproducibility_audit_path")
        if not isinstance(reproducibility_path, str) or not reproducibility_path.strip():
            run_errors.append("reproducibility_audit_path_missing")
        else:
            resolved_repro = Path(reproducibility_path).expanduser()
            if not resolved_repro.is_absolute():
                resolved_repro = path.parent / resolved_repro
            try:
                reproducibility = _read_manifest(resolved_repro.resolve())
                if reproducibility.get("status") != "PASS" or reproducibility.get("errors") != []:
                    run_errors.append("reproducibility_audit_not_clean_pass")
                if reproducibility.get("pipeline_manifest_path") and Path(
                    str(reproducibility["pipeline_manifest_path"])
                ).expanduser().resolve() != path:
                    run_errors.append("reproducibility_manifest_path_mismatch")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                run_errors.append(f"reproducibility_audit_unreadable:{type(exc).__name__}")

        completion_path = path.parent / f"{run_id}_pipeline_completion_audit.json"
        try:
            completion = _read_manifest(completion_path)
            if completion.get("run_id") != run_id:
                run_errors.append("completion_receipt_run_id_mismatch")
            if completion.get("status") != "PASS" or completion.get("errors") != [] or completion.get("warnings") != []:
                run_errors.append("completion_receipt_not_clean_pass")
            if completion.get("pipeline_manifest_sha256") != manifest.get("pipeline_manifest_sha256"):
                run_errors.append("completion_receipt_manifest_hash_mismatch")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            run_errors.append(f"completion_receipt_unreadable:{type(exc).__name__}")

        dataset_hash = manifest.get("dataset_sha256")
        if isinstance(dataset_hash, str) and len(dataset_hash) == 64:
            dataset_fingerprints.setdefault(dataset_hash.lower(), []).append(run_id)
        else:
            run_errors.append("dataset_sha256_invalid")

        run_results.append({
            "run_id": run_id,
            "manifest_path": str(path),
            "manifest_sha256": manifest_hash,
            "integrity_status": integrity.get("status"),
            "integrity_errors": integrity.get("errors", []),
            "integrity_warnings": integrity.get("warnings", []),
            "random_seed": seed,
            "status": "PASS" if not run_errors else "FAIL",
            "errors": sorted(set(run_errors)),
            "actionable": False,
            "execution_connected": False,
        })

    for field, values in protocol_values.items():
        if len(values) > 1:
            errors.append({
                "code": "research_protocol_mismatch",
                "field": field,
                "distinct_values": sorted(values),
            })
    if len(dataset_fingerprints) > 1:
        errors.append({
            "code": "dataset_fingerprint_mismatch",
            "dataset_fingerprints": sorted(dataset_fingerprints),
        })

    for run in run_results:
        if run["errors"]:
            errors.append({
                "code": "run_not_qualified",
                "run_id": run["run_id"],
                "errors": run["errors"],
            })

    runs_validated = sum(run["status"] == "PASS" for run in run_results)
    if runs_validated != len(run_results):
        status = "FAIL"
    elif errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return {
        "schema_version": 1,
        "audit_name": "research_experiment_registry",
        "status": status,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "runs_supplied": len(manifest_paths),
        "runs_validated": runs_validated,
        "run_results": run_results,
        "dataset_fingerprints": {
            digest: sorted(set(run_ids)) for digest, run_ids in sorted(dataset_fingerprints.items())
        },
        "protocol_fields_checked": list(_PROTOCOL_FIELDS),
        "protocol_mismatches": [
            field for field, values in protocol_values.items() if len(values) > 1
        ],
        "random_seeds_distinct": len(seen_seeds) == len(run_results),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "This checks recorded manifests and linked artifact integrity; it does not rerun experiments or recalculate metrics.",
            "A clean registry establishes protocol and artifact consistency only, not model superiority, profitability, statistical significance, or absence of all leakage.",
            "The registry never selects a model or authorizes deployment or trading.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="Run manifest path; supply at least two distinct runs")
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    inputs = {path.expanduser().resolve() for path in args.manifest}
    if output in inputs:
        print("output_must_not_alias_input", file=sys.stderr)
        return 2
    try:
        report = build_experiment_registry(args.manifest)
        _atomic_write_no_clobber(output, report)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_path": str(output),
        "runs_supplied": report["runs_supplied"],
        "runs_validated": report["runs_validated"],
        "error_count": len(report["errors"]),
        "execution_connected": False,
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

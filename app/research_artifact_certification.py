"""Offline end-to-end certification of research pipeline artifacts.

Checks artifact hash linkage, manifest self-hash, non-actionable flags, and dataset
chronology/label maturity. This is a structural audit, not a guarantee of profitability
or proof that all upstream feature code is leakage-free.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from app.research_reproducibility_auditor import audit_experiment_reproducibility


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve(base: Path, value: object, field: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"missing_artifact_path:{field}")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        errors.append(f"artifact_missing:{field}")
        return None
    return path


def _parse_aware_timestamp(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _audit_dataset(path: Path, errors: list[str]) -> dict[str, int]:
    rows = 0
    invalid_timestamps = 0
    non_chronological = 0
    immature_labels = 0
    previous: datetime | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "label_available_at", "label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            errors.append("dataset_missing_columns:" + ",".join(sorted(missing)))
            return {"rows": 0, "invalid_timestamps": 0, "non_chronological": 0, "immature_labels": 0}
        for line, row in enumerate(reader, start=2):
            rows += 1
            try:
                timestamp = _parse_aware_timestamp(row.get("timestamp") or "")
                available = _parse_aware_timestamp(row.get("label_available_at") or "")
            except (TypeError, ValueError):
                invalid_timestamps += 1
                continue
            if previous is not None and timestamp <= previous:
                non_chronological += 1
            previous = timestamp
            if available <= timestamp:
                immature_labels += 1
            try:
                label = int(row.get("label") or "")
                if label not in (0, 1):
                    raise ValueError("label must be binary")
            except (TypeError, ValueError):
                errors.append("dataset_invalid_binary_label")
                break
    if rows == 0:
        errors.append("dataset_has_no_rows")
    if invalid_timestamps:
        errors.append("dataset_invalid_or_timezone_naive_timestamps")
    if non_chronological:
        errors.append("dataset_timestamps_not_strictly_chronological")
    if immature_labels:
        errors.append("dataset_label_maturity_not_after_sample_time")
    return {
        "rows": rows,
        "invalid_timestamps": invalid_timestamps,
        "non_chronological": non_chronological,
        "immature_labels": immature_labels,
    }


def certify_pipeline_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate hashes and temporal invariants for a completed research pipeline run."""
    manifest_file = Path(manifest_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(manifest_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1, "status": "FAIL",
            "errors": [f"pipeline_manifest_unreadable:{type(exc).__name__}"],
            "warnings": [], "actionable": False, "execution_connected": False,
        }

    claimed_manifest_hash = manifest.get("pipeline_manifest_sha256")
    canonical_payload = dict(manifest)
    canonical_payload.pop("pipeline_manifest_sha256", None)
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    actual_manifest_hash = hashlib.sha256(canonical).hexdigest()
    if claimed_manifest_hash != actual_manifest_hash:
        errors.append("pipeline_manifest_sha256_mismatch")
    if manifest.get("actionable") is not False:
        errors.append("pipeline_actionable_must_be_false")
    if manifest.get("execution_connected") is not False:
        errors.append("pipeline_execution_connected_must_be_false")
    if manifest.get("automatic_winner_selection") is not False:
        errors.append("pipeline_automatic_winner_selection_must_be_false")
    if manifest.get("test_metrics_used_for_selection") is not False:
        errors.append("pipeline_test_metrics_used_for_selection_must_be_false")

    base = manifest_file.parent
    artifacts = {
        "raw_ohlc": ("raw_ohlc_path", "raw_ohlc_sha256"),
        "dataset": ("dataset_path", "dataset_sha256"),
        "dataset_manifest": ("dataset_manifest_path", "dataset_manifest_sha256"),
        "data_quality_audit": ("data_quality_audit_path", "data_quality_audit_sha256"),
        "experiment_matrix_report": ("experiment_matrix_report", "experiment_matrix_report_sha256"),
        "provenance_audit": ("provenance_audit_path", "provenance_audit_sha256"),
    }
    resolved: dict[str, Path] = {}
    for name, (path_field, hash_field) in artifacts.items():
        artifact = _resolve(base, manifest.get(path_field), path_field, errors)
        if artifact is None:
            continue
        resolved[name] = artifact
        if not isinstance(manifest.get(hash_field), str) or _sha256(artifact) != manifest[hash_field]:
            errors.append(f"artifact_sha256_mismatch:{name}")

    if "dataset" in resolved:
        dataset_stats = _audit_dataset(resolved["dataset"], errors)
    else:
        dataset_stats = {"rows": 0, "invalid_timestamps": 0, "non_chronological": 0, "immature_labels": 0}

    if "dataset_manifest" in resolved:
        try:
            dataset_manifest = _read_json(resolved["dataset_manifest"])
            if "raw_ohlc" in resolved and dataset_manifest.get("input_snapshot_sha256") != _sha256(resolved["raw_ohlc"]):
                errors.append("dataset_manifest_raw_snapshot_hash_mismatch")
            if "dataset" in resolved and dataset_manifest.get("output_dataset_sha256") != _sha256(resolved["dataset"]):
                errors.append("dataset_manifest_output_hash_mismatch")
            claimed = dataset_manifest.get("manifest_sha256")
            canonical_dataset = dict(dataset_manifest)
            canonical_dataset.pop("manifest_sha256", None)
            digest = hashlib.sha256(json.dumps(canonical_dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            if claimed != digest:
                errors.append("dataset_manifest_self_hash_mismatch")
            if dataset_manifest.get("actionable") is not False:
                errors.append("dataset_manifest_actionable_must_be_false")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("dataset_manifest_unreadable")

    if "data_quality_audit" in resolved:
        try:
            receipt = _read_json(resolved["data_quality_audit"])
            if "raw_ohlc" in resolved and receipt.get("source_sha256") != _sha256(resolved["raw_ohlc"]):
                errors.append("data_quality_source_hash_mismatch")
            if receipt.get("status") == "FAIL":
                errors.append("data_quality_audit_failed")
            if receipt.get("actionable") is not False:
                errors.append("data_quality_audit_actionable_must_be_false")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("data_quality_receipt_unreadable")

    if "provenance_audit" in resolved:
        try:
            receipt = _read_json(resolved["provenance_audit"])
            if "experiment_matrix_report" in resolved and receipt.get("source_report_sha256") != _sha256(resolved["experiment_matrix_report"]):
                errors.append("provenance_source_hash_mismatch")
            if receipt.get("status") == "FAIL":
                errors.append("provenance_audit_failed")
            if receipt.get("actionable") is not False:
                errors.append("provenance_audit_actionable_must_be_false")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("provenance_receipt_unreadable")

    fingerprint_path = _resolve(base, manifest.get("experiment_fingerprint_path"), "experiment_fingerprint_path", errors)
    if fingerprint_path is not None:
        reproducibility = audit_experiment_reproducibility(fingerprint_path, manifest_file)
        if reproducibility.get("status") != "PASS":
            errors.append("reproducibility_audit_failed")
            errors.extend("reproducibility:" + str(item) for item in reproducibility.get("errors", []))
    else:
        reproducibility = {"status": "FAIL", "errors": ["experiment_fingerprint_missing"]}

    if manifest.get("data_quality_audit_status") == "PASS_WITH_WARNINGS":
        warnings.append("data_quality_audit_has_warnings_review_session_gaps")
    elif manifest.get("data_quality_audit_status") != "PASS":
        errors.append("pipeline_data_quality_status_not_pass")
    if manifest.get("provenance_audit_status") == "PASS_WITH_WARNINGS":
        warnings.append("provenance_audit_has_warnings")
    elif manifest.get("provenance_audit_status") != "PASS":
        errors.append("pipeline_provenance_status_not_pass")

    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "pipeline_manifest_path": str(manifest_file),
        "pipeline_manifest_sha256_expected": claimed_manifest_hash,
        "pipeline_manifest_sha256_actual": actual_manifest_hash,
        "artifact_checks": {name: "CHECKED" if name in resolved else "MISSING" for name in artifacts},
        "dataset_temporal_checks": dataset_stats,
        "reproducibility_audit_status": reproducibility.get("status"),
        "reproducibility_audit_errors": reproducibility.get("errors", []),
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "End-to-end artifact integrity and label timestamp checks only. This receipt does not "
            "prove that upstream features are leakage-free, that source data is licensed, or that "
            "any model is profitable."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Certify hashes and temporal integrity for an offline research run.")
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = certify_pipeline_manifest(args.pipeline_manifest)
    target = Path(args.receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

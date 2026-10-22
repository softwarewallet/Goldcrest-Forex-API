"""Offline point-in-time feature provenance auditor for prepared research datasets.

This validates dataset/manifest integrity, feature schema, chronological rows, and
label maturity. It cannot prove implementation-level causality for arbitrary code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_CAUSAL_RULE = "Each feature at t uses only OHLC observations timestamped <= t."


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: str, field: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def audit_feature_provenance(dataset_csv: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    dataset_path, manifest_file = Path(dataset_csv), Path(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1, "status": "FAIL", "errors": [f"manifest_unreadable:{type(exc).__name__}"],
            "warnings": [], "actionable": False, "execution_connected": False,
        }
    if not isinstance(manifest, dict):
        manifest = {}
        errors.append("manifest_must_be_object")

    declared_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if not isinstance(declared_hash, str) or declared_hash != hashlib.sha256(_canonical(unhashed)).hexdigest():
        errors.append("manifest_sha256_mismatch")
    if manifest.get("actionable") is not False:
        errors.append("manifest_actionable_must_be_false")
    if manifest.get("automatic_sorting") is not False:
        errors.append("automatic_sorting_must_be_false")
    if manifest.get("automatic_imputation") is not False:
        errors.append("automatic_imputation_must_be_false")
    if manifest.get("feature_timestamp_rule") != _CAUSAL_RULE:
        errors.append("feature_timestamp_rule_missing_or_unrecognized")

    try:
        dataset_hash = _sha256(dataset_path)
    except OSError:
        dataset_hash = ""
        errors.append("dataset_unreadable")
    if not dataset_hash or dataset_hash != manifest.get("output_dataset_sha256"):
        errors.append("dataset_sha256_mismatch")

    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names or not all(isinstance(n, str) and n.strip() for n in feature_names):
        errors.append("feature_names_invalid")
        feature_names = []
    elif len(feature_names) != len(set(feature_names)):
        errors.append("feature_names_duplicate")

    row_count = 0
    if dataset_hash:
        try:
            with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                required = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close", "label", "label_available_at"}
                if not required.issubset(columns):
                    errors.append("dataset_required_columns_missing")
                if not set(feature_names).issubset(columns):
                    errors.append("manifest_feature_columns_missing_from_dataset")
                expected_features = [c for c in columns if c not in required]
                if feature_names != expected_features:
                    errors.append("manifest_feature_schema_mismatch")
                previous: datetime | None = None
                for row_num, row in enumerate(reader, start=2):
                    row_count += 1
                    try:
                        timestamp = _timestamp(row.get("timestamp", ""), f"row_{row_num}_timestamp")
                        available = _timestamp(row.get("label_available_at", ""), f"row_{row_num}_label_available_at")
                        if previous is not None and timestamp <= previous:
                            errors.append("dataset_timestamps_not_strictly_increasing")
                            break
                        previous = timestamp
                        if available <= timestamp:
                            errors.append(f"row_{row_num}_label_not_matured_after_timestamp")
                        label = int(row.get("label", ""))
                        if label not in (0, 1) or str(label) != row.get("label", "").strip():
                            errors.append(f"row_{row_num}_label_not_binary")
                        for name in feature_names:
                            raw = row.get(name, "")
                            if raw is None or not raw.strip():
                                errors.append(f"row_{row_num}_feature_missing:{name}")
                                continue
                            value = float(raw)
                            if not math.isfinite(value):
                                errors.append(f"row_{row_num}_feature_non_finite:{name}")
                    except (TypeError, ValueError, OverflowError):
                        errors.append(f"row_{row_num}_invalid_value")
                        continue
        except (OSError, csv.Error, UnicodeError):
            errors.append("dataset_csv_unreadable")
    if row_count == 0:
        errors.append("dataset_has_no_rows")

    warnings.append("Structural checks cannot prove feature implementation causality or exclude all leakage; review feature code and point-in-time data availability.")
    return {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS",
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "dataset_sha256": dataset_hash or None,
        "manifest_sha256": declared_hash if isinstance(declared_hash, str) else None,
        "row_count": row_count,
        "feature_count": len(feature_names),
        "actionable": False,
        "execution_connected": False,
        "interpretation": "Dataset/manifest integrity and declared point-in-time constraints only; not proof of causal implementation or profitability.",
    }


def audit_feature_provenance_file(dataset_csv: str | Path, manifest_path: str | Path, receipt_path: str | Path) -> dict[str, Any]:
    result = audit_feature_provenance(dataset_csv, manifest_path)
    result["source_dataset_sha256"] = _sha256(Path(dataset_csv)) if Path(dataset_csv).is_file() else None
    target = Path(receipt_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result["receipt_sha256"] = _sha256(target)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit point-in-time provenance for an offline research dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_feature_provenance_file(args.dataset, args.manifest, args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

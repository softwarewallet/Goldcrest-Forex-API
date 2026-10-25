"""Research-only comparison of TA-Lib and NumPy causal feature backends.

This audit is descriptive. It does not train models, select a backend automatically,
or connect to execution. Backend differences can change model inputs across environments.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from app.causal_ta_features import build_causal_ohlc_features
from app.evaluation import ForexBar


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_numpy_fallback(bars: Sequence[ForexBar]):
    """Call the production feature builder while making its optional TA-Lib import unavailable."""
    original_import = builtins.__import__

    def without_talib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "talib" or name.startswith("talib."):
            raise ImportError("TA-Lib disabled for backend parity audit")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = without_talib
    try:
        return build_causal_ohlc_features(bars)
    finally:
        builtins.__import__ = original_import


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit_feature_backend_parity(
    bars: Sequence[ForexBar],
    *,
    source_data_sha256: str,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    if (not isinstance(source_data_sha256, str) or len(source_data_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in source_data_sha256)):
        raise ValueError("source_data_sha256_invalid")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance_must_be_finite_and_nonnegative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance_must_be_finite_and_nonnegative")
    if not bars:
        raise ValueError("bars_must_not_be_empty")

    native = build_causal_ohlc_features(bars)
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_name": "feature_backend_parity",
        "dataset_rows": len(bars),
        "dataset_sha256": source_data_sha256,
        "native_backend": native.indicator_backend,
        "comparison_backend": "numpy_fallback",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "feature_results": {},
        "errors": [],
        "warnings": [
            "This compares feature implementations, not downstream model performance.",
            "A passing comparison does not establish temporal integrity or predictive validity.",
        ],
        "actionable": False,
        "automatic_model_selection": False,
        "inferential": False,
        "execution_connected": False,
        "model_training_performed": False,
    }
    if native.indicator_backend != "talib":
        report.update({
            "status": "NOT_COMPARABLE",
            "comparison_performed": False,
            "warnings": report["warnings"] + [
                "TA-Lib is unavailable in the active environment; native backend is already NumPy fallback."
            ],
        })
        return report

    fallback = _build_numpy_fallback(bars)
    if native.feature_names != fallback.feature_names:
        report["errors"].append("feature_name_set_mismatch")
    for name in native.feature_names:
        native_values = [row.get(name) for row in native.rows]
        fallback_values = [row.get(name) for row in fallback.rows]
        mask_mismatches = 0
        common: list[tuple[float, float]] = []
        native_only = 0
        fallback_only = 0
        for left, right in zip(native_values, fallback_values):
            left_ok, right_ok = _finite(left), _finite(right)
            if left_ok != right_ok:
                mask_mismatches += 1
                native_only += int(left_ok)
                fallback_only += int(right_ok)
            elif left_ok and right_ok:
                common.append((float(left), float(right)))
        absolute_errors = [abs(left - right) for left, right in common]
        relative_errors = [
            abs(left - right) / max(abs(left), abs(right), absolute_tolerance, 1e-15)
            for left, right in common
        ]
        failed_values = sum(
            1 for left, right in common
            if abs(left - right) > absolute_tolerance + relative_tolerance * abs(right)
        )
        feature_passed = mask_mismatches == 0 and failed_values == 0
        report["feature_results"][name] = {
            "native_finite_rows": sum(_finite(value) for value in native_values),
            "fallback_finite_rows": sum(_finite(value) for value in fallback_values),
            "common_finite_rows": len(common),
            "finite_mask_mismatch_rows": mask_mismatches,
            "native_only_finite_rows": native_only,
            "fallback_only_finite_rows": fallback_only,
            "max_absolute_error": max(absolute_errors) if absolute_errors else None,
            "mean_absolute_error": sum(absolute_errors) / len(absolute_errors) if absolute_errors else None,
            "max_relative_error": max(relative_errors) if relative_errors else None,
            "values_outside_tolerance": failed_values,
            "passed": feature_passed,
        }
        if not feature_passed:
            report["errors"].append(f"backend_parity_failed:{name}")
    report["comparison_performed"] = True
    report["status"] = "PASS_WITH_WARNINGS" if not report["errors"] else "FAIL"
    report["all_features_passed"] = bool(report["feature_results"]) and all(
        item["passed"] for item in report["feature_results"].values()
    )
    return report


def _read_bars(path: Path, default_symbol: str, default_timeframe: str) -> tuple[list[ForexBar], str]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".csv":
        import csv
        try:
            records = [
                {str(key).strip().lower(): value for key, value in row.items() if key is not None}
                for row in csv.DictReader(raw.decode("utf-8-sig").splitlines())
            ]
        except UnicodeDecodeError as exc:
            raise ValueError("input_csv_invalid") from exc
    else:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("input_json_invalid") from exc
        records = payload.get("bars") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("input_bars_missing")
    bars: list[ForexBar] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"bar_record_invalid:{index}")
        stamp = next((row[key] for key in ("timestamp", "timestamp_utc", "datetime", "time") if key in row), None)
        try:
            if not isinstance(stamp, str):
                raise TypeError
            timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            bars.append(ForexBar(
                symbol=str(row.get("symbol") or default_symbol),
                timeframe=str(row.get("timeframe") or default_timeframe),
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"bar_record_schema_invalid:{index}") from exc
    return bars, _sha256(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV or JSON OHLC file")
    parser.add_argument("--symbol", default="EUR/USD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_feature_backend_parity_audit.json")
    try:
        bars, digest = _read_bars(args.input, args.symbol, args.timeframe)
        report = audit_feature_backend_parity(
            bars,
            source_data_sha256=digest,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "dataset_rows": report["dataset_rows"],
            "native_backend": report["native_backend"],
            "comparison_performed": report["comparison_performed"],
            "all_features_passed": report.get("all_features_passed"),
            "actionable": False,
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] in {"PASS_WITH_WARNINGS", "NOT_COMPARABLE"} else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

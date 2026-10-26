"""Metamorphic point-in-time invariance audit for causal OHLC feature engineering.

Recomputes prefixes of an immutable OHLC history and verifies that feature values already
available at each prefix are unchanged when later observations are added. Offline only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from app.causal_ta_features import build_causal_ohlc_features
from app.evaluation import ForexBar

_REQUIRED = {"timestamp", "open", "high", "low", "close"}
_FALSE_FLAGS = {
    "actionable": False,
    "execution_connected": False,
    "model_training_performed": False,
    "automatic_model_selection": False,
}


def _parse_time(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp_must_include_timezone")
    return parsed


def _load_bars(path: Path, symbol: str, timeframe: str) -> list[ForexBar]:
    bars: list[ForexBar] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED - columns
        if missing:
            raise ValueError(f"ohlc_columns_missing:{','.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                row_symbol = (row.get("symbol") or symbol).strip().upper()
                row_timeframe = (row.get("timeframe") or timeframe).strip().upper()
                if row_symbol != symbol or row_timeframe != timeframe:
                    raise ValueError("symbol_or_timeframe_mismatch")
                bars.append(ForexBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=_parse_time(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                ))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid_ohlc_row:{line}:{exc}") from exc
    if len(bars) < 2:
        raise ValueError("at_least_two_ohlc_rows_required")
    return bars


def _same_value(left: Any, right: Any, *, rel_tol: float, abs_tol: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return left == right
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def audit_point_in_time_invariance(
    ohlc_csv: str | Path,
    *,
    symbol: str,
    timeframe: str,
    max_cutoffs: int = 8,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
) -> dict[str, Any]:
    """Check that adding future OHLC bars never changes previously computed feature rows."""
    source = Path(ohlc_csv).expanduser().resolve()
    symbol, timeframe = symbol.strip().upper(), timeframe.strip().upper()
    if not symbol or not timeframe:
        raise ValueError("symbol_and_timeframe_required")
    if isinstance(max_cutoffs, bool) or not isinstance(max_cutoffs, int) or not 1 <= max_cutoffs <= 64:
        raise ValueError("max_cutoffs_must_be_integer_between_1_and_64")
    if not math.isfinite(rel_tol) or not math.isfinite(abs_tol) or rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances_must_be_finite_and_nonnegative")

    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    bars = _load_bars(source, symbol, timeframe)
    full = build_causal_ohlc_features(bars)
    last_index = len(bars) - 1
    cutoff_indexes = sorted({
        max(1, round((slot + 1) * last_index / min(max_cutoffs, last_index)))
        for slot in range(min(max_cutoffs, last_index))
    })
    errors: list[dict[str, Any]] = []
    compared = 0
    for cutoff in cutoff_indexes:
        prefix = build_causal_ohlc_features(bars[:cutoff + 1])
        if prefix.feature_names != full.feature_names:
            errors.append({"code": "feature_schema_changed_with_future_data", "cutoff_row": cutoff})
            continue
        for row_index in range(cutoff + 1):
            for feature in full.feature_names:
                compared += 1
                if not _same_value(
                    full.rows[row_index].get(feature),
                    prefix.rows[row_index].get(feature),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    errors.append({
                        "code": "future_data_changed_past_feature",
                        "cutoff_row": cutoff,
                        "affected_row": row_index,
                        "feature": feature,
                    })
                    if len(errors) >= 100:
                        break
            if len(errors) >= 100:
                break
        if len(errors) >= 100:
            break

    status = "FAIL" if errors else "PASS"
    return {
        "schema_version": 1,
        "audit_name": "research_point_in_time_feature_invariance_audit",
        "scope": "OFFLINE_OHLC_PREFIX_INVARIANCE",
        "status": status,
        "source_path": str(source),
        "source_sha256": raw_hash,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(bars),
        "feature_count": len(full.feature_names),
        "feature_names": list(full.feature_names),
        "indicator_backend": full.indicator_backend,
        "cutoff_rows_checked": cutoff_indexes,
        "cutoff_count": len(cutoff_indexes),
        "feature_values_compared": compared,
        "relative_tolerance": rel_tol,
        "absolute_tolerance": abs_tol,
        "errors": errors,
        "warnings": [
            "Prefix invariance detects future-data dependence in computed features but does not prove that every feature is economically valid or that labels/folds are leakage-free."
        ],
        **_FALSE_FLAGS,
    }


def _write_no_clobber(path: Path, report: dict[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlc-csv", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cutoffs", type=int, default=8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    try:
        source = args.ohlc_csv.expanduser().resolve()
        output = args.output.expanduser().resolve()
        if output == source:
            raise ValueError("output_must_not_alias_ohlc_input")
        report = audit_point_in_time_invariance(
            source, symbol=args.symbol, timeframe=args.timeframe,
            max_cutoffs=args.max_cutoffs, rel_tol=args.relative_tolerance,
            abs_tol=args.absolute_tolerance,
        )
        published = _write_no_clobber(output, report)
        print(json.dumps({
            "status": report["status"],
            "report_path": str(published),
            "rows": report["rows"],
            "cutoff_count": report["cutoff_count"],
            "feature_values_compared": report["feature_values_compared"],
            "error_count": len(report["errors"]),
            "actionable": False,
        }, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

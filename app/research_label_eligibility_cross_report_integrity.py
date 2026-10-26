"""Integrity wrapper for multi-cap label-eligibility comparison reports.

Captures exact input hashes and provenance identities before and after the common-cohort
audit. It fails closed if any input changes while the audit is running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from app.research_label_eligibility_common_cohort import audit_common_heldout_cohort


def _snapshot(path: str) -> dict:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "baseline_run_id": value.get("baseline_run_id"),
        "variant_run_id": value.get("variant_run_id"),
        "max_label_elapsed_hours": value.get("max_label_elapsed_hours"),
        "symbol": value.get("symbol"),
        "timeframe": value.get("timeframe"),
        "label_horizon_bars": value.get("label_horizon_bars"),
        "raw_ohlc_sha256": value.get("raw_ohlc_sha256"),
        "baseline_dataset_sha256": value.get("baseline_dataset_sha256"),
    }


def audit_cross_report_integrity(
    report_paths: Sequence[str],
    *,
    allow_baseline_id_variation: bool = False,
) -> dict:
    try:
        if len(report_paths) < 2:
            raise ValueError("provide at least two comparison reports")
        before = [_snapshot(path) for path in report_paths]
        cohort = audit_common_heldout_cohort(
            report_paths,
            allow_baseline_id_variation=allow_baseline_id_variation,
        )
        after = [_snapshot(path) for path in report_paths]
        if before != after:
            raise ValueError("input comparison report changed during audit")
        if cohort.get("status") != "PASS":
            raise ValueError("common-cohort audit failed: " + "; ".join(cohort.get("errors", [])))
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_CROSS_REPORT_INTEGRITY",
            "status": "PASS",
            "input_report_receipts": before,
            "cross_report_audit": cohort,
            "input_count": len(before),
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
            "interpretation": (
                "Input byte hashes and provenance identities are recorded. The embedded "
                "common-cohort audit is descriptive only and does not prove model validity, "
                "identical training, or profitability."
            ),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_CROSS_REPORT_INTEGRITY",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-report", action="append", required=True)
    parser.add_argument("--allow-baseline-id-variation-with-matching-dataset-hash", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_cross_report_integrity(
        args.comparison_report,
        allow_baseline_id_variation=args.allow_baseline_id_variation_with_matching_dataset_hash,
    )
    output = Path(args.output).expanduser().resolve()
    inputs = {Path(path).expanduser().resolve() for path in args.comparison_report}
    if output in inputs:
        print(json.dumps({"status": "FAIL", "error": "output_must_not_alias_input",
                          "actionable": False, "execution_connected": False}), file=sys.stderr)
        return 2
    if output.exists():
        print(json.dumps({"status": "FAIL", "error": "output_already_exists",
                          "actionable": False, "execution_connected": False}), file=sys.stderr)
        return 2
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\\n")
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}",
                          "actionable": False, "execution_connected": False}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "output": str(output),
                      "input_count": report.get("input_count", 0),
                      "actionable": False}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

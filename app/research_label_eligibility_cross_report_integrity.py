"""Integrity wrapper for multi-cap label-eligibility comparison reports.

Captures exact input hashes and provenance identities before and after the common-cohort
audit. It fails closed if any input changes while the audit is running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from app.research_label_eligibility_common_cohort import (
    _reject_duplicate_json_keys,
    audit_common_heldout_cohort,
)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard_json_constant:{value}")


def _snapshot(path: str) -> dict:
    raw = Path(path).read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
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
        resolved_paths = [str(Path(item["path"]).expanduser().resolve()) for item in before]
        if len(set(resolved_paths)) != len(resolved_paths):
            raise ValueError("duplicate_input_report_path")
        variant_ids = [item.get("variant_run_id") for item in before]
        if any(not isinstance(item, str) or not item.strip() for item in variant_ids):
            raise ValueError("invalid_or_missing_variant_run_id")
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("duplicate_variant_run_id")
        for item in before:
            for field in ("baseline_run_id", "symbol", "timeframe"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid_or_missing_{field}")
            for field in ("raw_ohlc_sha256", "baseline_dataset_sha256"):
                digest = item.get(field)
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdefABCDEF" for char in digest
                ):
                    raise ValueError(f"invalid_or_missing_{field}")
        cohort = audit_common_heldout_cohort(
            report_paths,
            allow_baseline_id_variation=allow_baseline_id_variation,
        )
        after = [_snapshot(path) for path in report_paths]
        if before != after:
            raise ValueError("input comparison report changed during audit")
        cohort_receipts = cohort.get("input_report_receipts")
        if cohort.get("status") == "PASS":
            if not isinstance(cohort_receipts, list) or [
                item.get("sha256") for item in cohort_receipts if isinstance(item, dict)
            ] != [item["sha256"] for item in before]:
                raise ValueError("input snapshot receipts do not match wrapper snapshots")
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


def _write_report_no_clobber(output_path: str | Path, payload: dict, input_paths: Sequence[str]) -> Path:
    """Atomically publish a complete JSON report without overwriting inputs or destinations."""
    output = Path(output_path).expanduser().resolve()
    inputs = {Path(path).expanduser().resolve() for path in input_paths}
    if output in inputs:
        raise ValueError("output_must_not_alias_input")
    if output.exists():
        raise ValueError(f"output_already_exists:{output}")

    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    published_identity: tuple[int, int] | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        # Hard-link publication fails atomically if a competing writer wins.
        os.link(temporary, output)
        stat = output.stat()
        published_identity = (stat.st_dev, stat.st_ino)
        temporary.unlink()
        temporary = None
        return output
    except Exception:
        if published_identity is not None:
            try:
                current = output.stat()
                if (current.st_dev, current.st_ino) == published_identity:
                    output.unlink()
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_cross_report_integrity(
        args.comparison_report,
        allow_baseline_id_variation=args.allow_baseline_id_variation_with_matching_dataset_hash,
    )
    try:
        output = _write_report_no_clobber(args.output, report, args.comparison_report)
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

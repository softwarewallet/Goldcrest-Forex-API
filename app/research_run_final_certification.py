"""Run final evidence-bundle verification and certification-chain audit in one offline command.

This orchestrator consumes existing reports only. It does not execute experiments, rerun upstream
research audits, train/select models, connect to a broker, or authorize deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.research_evaluation_certification_evidence_bundle import verify_evidence_bundle
from app.research_final_certification_chain_audit import audit_final_certification_chain


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def run_final_certification(
    readiness: dict[str, Any],
    bundle: dict[str, Any],
    *,
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_evidence_bundle(bundle, evidence_root=evidence_root)
    final_report = audit_final_certification_chain(readiness, bundle, verification)
    return verification, final_report


def _write_pair_atomically(
    first_path: Path,
    first_payload: dict[str, Any],
    second_path: Path,
    second_payload: dict[str, Any],
) -> None:
    first_path = first_path.resolve()
    second_path = second_path.resolve()
    if first_path == second_path:
        raise ValueError("output_paths_must_be_distinct")
    if first_path.exists():
        raise ValueError(f"output_already_exists:{first_path}")
    if second_path.exists():
        raise ValueError(f"output_already_exists:{second_path}")
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, payload in (
            (first_path, first_payload),
            (second_path, second_payload),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                staged.append((Path(stream.name), destination))
        # Recheck before publishing so an existing output is never overwritten.
        if any(destination.exists() for _, destination in staged):
            raise ValueError("output_created_during_run")
        for temporary, destination in staged:
            os.replace(temporary, destination)
    except Exception:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--evidence-bundle", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--verification-output", required=True, type=Path)
    parser.add_argument("--final-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        readiness = _read_object(args.readiness_report)
        bundle = _read_object(args.evidence_bundle)
        verification, final_report = run_final_certification(
            readiness, bundle, evidence_root=args.evidence_root
        )
        _write_pair_atomically(
            args.verification_output, verification,
            args.final_output, final_report,
        )
        print(json.dumps({
            "status": final_report["status"],
            "readiness": final_report["readiness"],
            "bundle_verification_status": verification["status"],
            "reports_verified": verification["reports_verified"],
            "error_count": len(final_report["errors"]),
            "verification_report": str(args.verification_output.resolve()),
            "final_report": str(args.final_output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if final_report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL",
            "error": f"{type(exc).__name__}:{exc}",
            "deployment_authorized": False,
        }, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

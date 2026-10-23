"""Canonical reproducibility fingerprints for offline research experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_experiment_fingerprint(
    *,
    configuration: dict[str, Any],
    raw_ohlc_sha256: str,
    dataset_sha256: str,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint immutable data provenance plus every declared experiment parameter."""
    for field, digest in (("raw_ohlc_sha256", raw_ohlc_sha256), ("dataset_sha256", dataset_sha256)):
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    for field in ("feature_implementation_sha256", "feature_schema_sha256"):
        digest = dataset_manifest.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"dataset manifest {field} is missing or invalid")
    if dataset_manifest.get("actionable") is not False:
        raise ValueError("dataset manifest must be explicitly non-actionable")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_REPRODUCIBILITY",
        "configuration": configuration,
        "raw_ohlc_sha256": raw_ohlc_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": hashlib.sha256(canonical_json({k: v for k, v in dataset_manifest.items() if k != "manifest_sha256"})).hexdigest(),
        "feature_implementation_sha256": dataset_manifest["feature_implementation_sha256"],
        "feature_schema_sha256": dataset_manifest["feature_schema_sha256"],
        "feature_backend": dataset_manifest.get("feature_backend"),
        "actionable": False,
        "execution_connected": False,
    }
    payload["experiment_fingerprint_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def verify_experiment_fingerprint(receipt: dict[str, Any]) -> dict[str, Any]:
    """Check the canonical self-hash and non-actionable safety contract of a receipt."""
    errors: list[str] = []
    claimed = receipt.get("experiment_fingerprint_sha256")
    payload = dict(receipt)
    payload.pop("experiment_fingerprint_sha256", None)
    actual = hashlib.sha256(canonical_json(payload)).hexdigest()
    if claimed != actual:
        errors.append("experiment_fingerprint_sha256_mismatch")
    if receipt.get("actionable") is not False:
        errors.append("actionable_must_be_false")
    if receipt.get("execution_connected") is not False:
        errors.append("execution_connected_must_be_false")
    return {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "experiment_fingerprint_sha256": claimed if isinstance(claimed, str) else None,
        "actionable": False,
        "execution_connected": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify an offline research experiment fingerprint receipt.")
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be a JSON object")
        result = verify_experiment_fingerprint(receipt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "errors": [f"receipt_unreadable:{type(exc).__name__}"], "actionable": False, "execution_connected": False}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

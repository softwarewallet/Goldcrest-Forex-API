"""Cross-check offline experiment fingerprint and pipeline artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from app.research_experiment_fingerprint import canonical_json, verify_experiment_fingerprint


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact(value: Any, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute() and (manifest_path.parent / candidate).is_file():
        return manifest_path.parent / candidate
    return candidate


def audit_experiment_reproducibility(fingerprint_path: str | Path, pipeline_manifest_path: str | Path) -> dict[str, Any]:
    """Fail closed if linked fingerprint, pipeline, or dataset-manifest evidence disagrees."""
    fp_path, pipe_path = Path(fingerprint_path), Path(pipeline_manifest_path)
    errors: list[str] = []
    checks: dict[str, str] = {}
    try:
        fingerprint = _read_json(fp_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        fingerprint = {}
        errors.append(f"fingerprint_unreadable:{type(exc).__name__}")
    try:
        manifest = _read_json(pipe_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = {}
        errors.append(f"pipeline_manifest_unreadable:{type(exc).__name__}")

    if fingerprint:
        verification = verify_experiment_fingerprint(fingerprint)
        checks["fingerprint_self_hash"] = verification["status"]
        errors.extend(verification["errors"])
        if fingerprint.get("actionable") is not False:
            errors.append("fingerprint_actionable_must_be_false")
        if fingerprint.get("execution_connected") is not False:
            errors.append("fingerprint_execution_connected_must_be_false")

    if manifest:
        if manifest.get("actionable") is not False:
            errors.append("pipeline_actionable_must_be_false")
        if manifest.get("execution_connected") is not False:
            errors.append("pipeline_execution_connected_must_be_false")
        claimed = manifest.get("pipeline_manifest_sha256")
        payload = dict(manifest)
        payload.pop("pipeline_manifest_sha256", None)
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        checks["pipeline_manifest_self_hash"] = "PASS" if claimed == actual else "FAIL"
        if claimed != actual:
            errors.append("pipeline_manifest_sha256_mismatch")

    if fingerprint and manifest:
        claimed_fp = fingerprint.get("experiment_fingerprint_sha256")
        checks["pipeline_fingerprint_link"] = "PASS" if manifest.get("experiment_fingerprint_sha256") == claimed_fp else "FAIL"
        if manifest.get("experiment_fingerprint_sha256") != claimed_fp:
            errors.append("pipeline_fingerprint_hash_mismatch")
        actual_file_hash = _sha256(fp_path)
        checks["fingerprint_file_hash_link"] = "PASS" if manifest.get("experiment_fingerprint_file_sha256") == actual_file_hash else "FAIL"
        if manifest.get("experiment_fingerprint_file_sha256") != actual_file_hash:
            errors.append("pipeline_fingerprint_file_hash_mismatch")
        for key in ("raw_ohlc_sha256", "dataset_sha256"):
            checks[key] = "PASS" if fingerprint.get(key) == manifest.get(key) else "FAIL"
            if checks[key] == "FAIL":
                errors.append(f"{key}_mismatch")
        dataset_path = _resolve_artifact(manifest.get("dataset_manifest_path"), pipe_path)
        if dataset_path is None or not dataset_path.is_file():
            errors.append("dataset_manifest_missing")
            checks["dataset_manifest_link"] = "FAIL"
        else:
            try:
                dataset_manifest = _read_json(dataset_path)
                canonical_manifest = {k: v for k, v in dataset_manifest.items() if k != "manifest_sha256"}
                dataset_hash = hashlib.sha256(canonical_json(canonical_manifest)).hexdigest()
                matched = dataset_hash == fingerprint.get("dataset_manifest_sha256")
                matched = matched and dataset_manifest.get("feature_implementation_sha256") == fingerprint.get("feature_implementation_sha256")
                matched = matched and dataset_manifest.get("feature_schema_sha256") == fingerprint.get("feature_schema_sha256")
                checks["dataset_manifest_link"] = "PASS" if matched else "FAIL"
                if not matched:
                    errors.append("dataset_manifest_fingerprint_mismatch")
            except (OSError, ValueError, json.JSONDecodeError):
                errors.append("dataset_manifest_unreadable")
                checks["dataset_manifest_link"] = "FAIL"

    return {"schema_version": 1, "scope": "OFFLINE_RESEARCH_REPRODUCIBILITY_AUDIT", "status": "FAIL" if errors else "PASS", "errors": sorted(set(errors)), "checks": checks, "fingerprint_path": str(fp_path), "fingerprint_file_sha256": _sha256(fp_path) if fp_path.is_file() else None, "pipeline_manifest_path": str(pipe_path), "actionable": False, "execution_connected": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit offline experiment reproducibility artifacts.")
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_experiment_reproducibility(args.fingerprint, args.pipeline_manifest)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

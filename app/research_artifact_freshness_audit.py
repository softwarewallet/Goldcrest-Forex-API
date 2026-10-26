"""Offline freshness audit for research artifacts and dependency-based regeneration planning.

This tool compares recorded provenance with explicitly supplied current fingerprints. It never
regenerates artifacts, executes experiments, trains/selects models, connects to brokers, or trades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which are not valid RFC 8259 JSON."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fingerprint_path(path: Path) -> str:
    """Hash one file or a directory tree deterministically; reject symlinks and special files."""
    target = path.resolve(strict=True)
    if target.is_file():
        return _sha256(target.read_bytes())
    if not target.is_dir():
        raise ValueError(f"fingerprint_target_not_file_or_directory:{path}")
    digest = hashlib.sha256()
    files = sorted(target.rglob("*"), key=lambda item: item.relative_to(target).as_posix())
    for item in files:
        if item.is_symlink():
            raise ValueError(f"symlink_not_allowed_in_fingerprint:{item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"special_file_not_allowed_in_fingerprint:{item}")
        relative = item.relative_to(target).as_posix().encode("utf-8")
        raw = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _valid_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def audit_artifact_freshness(
    manifest: dict[str, Any],
    *,
    current_revision: str,
    current_config_sha256: str,
    current_dataset_sha256: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("audit_name") != "research_artifact_freshness_manifest":
        raise ValueError("freshness_manifest_invalid")
    if not isinstance(current_revision, str) or not REVISION_RE.fullmatch(current_revision):
        raise ValueError("current_revision_must_be_full_40_character_git_sha")
    if not _valid_fingerprint(current_config_sha256):
        raise ValueError("current_config_sha256_invalid")
    if not _valid_fingerprint(current_dataset_sha256):
        raise ValueError("current_dataset_sha256_invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest_artifacts_must_be_list")

    by_id: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("artifact_entry_must_be_object")
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id_required")
        if artifact_id in by_id:
            raise ValueError(f"duplicate_artifact_id:{artifact_id}")
        dependencies = item.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) for dep in dependencies):
            raise ValueError(f"dependencies_must_be_string_list:{artifact_id}")
        by_id[artifact_id] = item

    for artifact_id, item in by_id.items():
        for dependency in item.get("depends_on", []):
            if dependency not in by_id:
                raise ValueError(f"unknown_dependency:{artifact_id}:{dependency}")
            if dependency == artifact_id:
                raise ValueError(f"self_dependency:{artifact_id}")

    # Detect dependency cycles before classification so propagation is deterministic.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"dependency_cycle:{node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in by_id[node].get("depends_on", []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for artifact_id in by_id:
        visit(artifact_id)

    results: dict[str, dict[str, Any]] = {}
    for artifact_id, item in by_id.items():
        reasons: list[str] = []
        recorded_revision = item.get("source_revision")
        recorded_config = item.get("config_sha256")
        recorded_dataset = item.get("dataset_sha256")
        if not isinstance(recorded_revision, str) or not REVISION_RE.fullmatch(recorded_revision):
            reasons.append("MISSING_OR_INVALID_SOURCE_REVISION")
        elif recorded_revision.lower() != current_revision.lower():
            reasons.append("STALE_CODE_REVISION")
        if not _valid_fingerprint(recorded_config):
            reasons.append("MISSING_OR_INVALID_CONFIG_FINGERPRINT")
        elif recorded_config != current_config_sha256:
            reasons.append("STALE_CONFIG")
        if not _valid_fingerprint(recorded_dataset):
            reasons.append("MISSING_OR_INVALID_DATASET_FINGERPRINT")
        elif recorded_dataset != current_dataset_sha256:
            reasons.append("STALE_DATASET")
        if item.get("path") is None or not isinstance(item.get("path"), str) or not item.get("path").strip():
            reasons.append("MISSING_ARTIFACT_PATH")
        results[artifact_id] = {
            "artifact_id": artifact_id,
            "path": item.get("path"),
            "depends_on": list(item.get("depends_on", [])),
            "reasons": reasons,
            "needs_regeneration": bool(reasons),
        }

    # A dependent artifact must be rebuilt if any dependency needs regeneration, even if its own
    # fingerprints match. Iterate to a fixed point (cycles have already been rejected).
    changed = True
    while changed:
        changed = False
        for artifact_id, result in results.items():
            stale_dependencies = [
                dep for dep in result["depends_on"] if results[dep]["needs_regeneration"]
            ]
            if stale_dependencies and "STALE_DEPENDENCY" not in result["reasons"]:
                result["reasons"].append("STALE_DEPENDENCY")
                result["stale_dependencies"] = sorted(stale_dependencies)
                result["needs_regeneration"] = True
                changed = True
            elif stale_dependencies:
                result["stale_dependencies"] = sorted(stale_dependencies)

    for result in results.values():
        result["reasons"] = sorted(set(result["reasons"]))
        result["classification"] = "REGENERATION_REQUIRED" if result["needs_regeneration"] else "CURRENT"

    stale = [key for key, value in results.items() if value["needs_regeneration"]]
    status = "STALE" if stale else "PASS"
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_freshness_audit",
        "status": status,
        "readiness": "REGENERATION_REQUIRED" if stale else "ALL_DECLARED_ARTIFACTS_CURRENT",
        "current_source_revision": current_revision.lower(),
        "current_config_sha256": current_config_sha256,
        "current_dataset_sha256": current_dataset_sha256,
        "artifact_count": len(results),
        "current_count": len(results) - len(stale),
        "regeneration_required_count": len(stale),
        "artifacts": [results[key] for key in sorted(results)],
        "regeneration_order": _dependency_order(results, stale),
        "errors": [],
        **FALSE_FLAGS,
        "scope_limitations": [
            "The audit compares caller-supplied provenance; it does not prove that a claimed revision or fingerprint was recorded honestly.",
            "The caller must supply the exact intended source revision and canonical config/dataset targets; a passing comparison cannot compensate for an incomplete manifest.",
            "Regeneration is only recommended in dependency order. No artifact is modified and no experiment is run.",
            "A changed source revision marks artifacts stale conservatively; a future allowlist-based compatibility policy would need explicit, reviewed dependency contracts.",
            "This report does not establish strategy profitability, predictive validity, statistical significance, or deployment suitability.",
        ],
    }


def _dependency_order(results: dict[str, dict[str, Any]], stale_ids: list[str]) -> list[str]:
    stale_set = set(stale_ids)
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited or node not in stale_set:
            return
        visited.add(node)
        for dependency in results[node]["depends_on"]:
            visit(dependency)
        ordered.append(node)

    for node in sorted(stale_set):
        visit(node)
    return ordered


def _read_object(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--current-revision", required=True)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_artifact_freshness(
            _read_object(args.manifest),
            current_revision=args.current_revision,
            current_config_sha256=fingerprint_path(args.config_path),
            current_dataset_sha256=fingerprint_path(args.dataset_path),
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "artifact_count": report["artifact_count"],
            "current_count": report["current_count"],
            "regeneration_required_count": report["regeneration_required_count"],
            "regeneration_order": report["regeneration_order"],
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

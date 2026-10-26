"""Capture and verify content-addressed provenance for offline research artifacts.

Capture hashes existing outputs plus their source revision, config, and dataset fingerprints.
Verify checks those captured hashes against current bytes. No experiments are run and no outputs
are regenerated. The manifest itself should be retained separately from the files it describes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from app.research_strict_json import strict_json_loads
from app.research_atomic_output import write_text_no_clobber

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
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


def fingerprint_path(path: Path) -> tuple[str, int, str]:
    """Return (SHA-256, byte count, kind) for a file or deterministic directory tree."""
    if path.is_symlink():
        raise ValueError(f"symlink_not_allowed:{path}")
    target = path.resolve(strict=True)
    if target.is_file():
        raw = target.read_bytes()
        return hashlib.sha256(raw).hexdigest(), len(raw), "file"
    if not target.is_dir():
        raise ValueError(f"target_not_file_or_directory:{path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for item in sorted(target.rglob("*"), key=lambda p: p.relative_to(target).as_posix()):
        if item.is_symlink():
            raise ValueError(f"symlink_not_allowed:{item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"special_file_not_allowed:{item}")
        relative = item.relative_to(target).as_posix().encode("utf-8")
        raw = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        total_bytes += len(raw)
    return digest.hexdigest(), total_bytes, "directory"


def _validate_revision(value: Any) -> str:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        raise ValueError("source_revision_must_be_full_40_character_git_sha")
    return value


def capture_provenance(
    specification: dict[str, Any],
    *,
    source_revision: str,
    config_path: Path,
    dataset_path: Path,
    root: Path,
) -> dict[str, Any]:
    revision = _validate_revision(source_revision)
    config_hash, config_size, config_kind = fingerprint_path(config_path)
    dataset_hash, dataset_size, dataset_kind = fingerprint_path(dataset_path)
    entries = specification.get("artifacts") if isinstance(specification, dict) else None
    if not isinstance(entries, list):
        raise ValueError("specification_artifacts_must_be_list")

    root = root.resolve(strict=True)
    seen: set[str] = set()
    seen_paths: set[str] = set()
    captured: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("artifact_spec_must_be_object")
        artifact_id = entry.get("artifact_id")
        relative_path = entry.get("path")
        if not isinstance(artifact_id, str) or not artifact_id.strip() or artifact_id != artifact_id.strip():
            raise ValueError("artifact_id_must_be_nonempty_and_canonical")
        if artifact_id in seen:
            raise ValueError(f"duplicate_artifact_id:{artifact_id}")
        seen.add(artifact_id)
        if not isinstance(relative_path, str) or not relative_path.strip() or relative_path != relative_path.strip():
            raise ValueError(f"artifact_path_must_be_nonempty_and_canonical:{artifact_id}")
        supplied_posix = PurePosixPath(relative_path)
        if (
            supplied_posix.is_absolute()
            or ".." in supplied_posix.parts
            or "\\" in relative_path
            or supplied_posix.as_posix() != relative_path
            or relative_path in {".", ""}
        ):
            raise ValueError(f"artifact_path_must_be_safe_canonical_relative_path:{artifact_id}")
        if relative_path in seen_paths:
            raise ValueError(f"duplicate_artifact_path:{relative_path}")
        seen_paths.add(relative_path)
        supplied = Path(relative_path)
        candidate = root / supplied
        if candidate.is_symlink():
            raise ValueError(f"symlink_not_allowed:{candidate}")
        resolved = candidate.resolve(strict=True)
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"artifact_path_outside_root:{artifact_id}")
        digest, size, kind = fingerprint_path(resolved)
        dependencies = entry.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(dep, str) or not dep.strip() or dep != dep.strip()
            for dep in dependencies
        ):
            raise ValueError(f"dependencies_must_be_canonical_string_list:{artifact_id}")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"duplicate_dependency:{artifact_id}")
        captured.append({
            "artifact_id": artifact_id,
            "path": supplied.as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "kind": kind,
            "source_revision": revision,
            "config_sha256": config_hash,
            "config_size_bytes": config_size,
            "config_kind": config_kind,
            "dataset_sha256": dataset_hash,
            "dataset_size_bytes": dataset_size,
            "dataset_kind": dataset_kind,
            "depends_on": dependencies,
        })

    ids = {item["artifact_id"] for item in captured}
    for item in captured:
        for dep in item["depends_on"]:
            if dep not in ids:
                raise ValueError(f"unknown_dependency:{item['artifact_id']}:{dep}")
            if dep == item["artifact_id"]:
                raise ValueError(f"self_dependency:{dep}")
    _assert_acyclic(captured)
    captured.sort(key=lambda item: item["artifact_id"])
    payload = {
        "schema_version": 1,
        "audit_name": "research_artifact_provenance_manifest",
        "source_revision": revision,
        "config": {"path": str(config_path.resolve()), "sha256": config_hash, "size_bytes": config_size, "kind": config_kind},
        "dataset": {"path": str(dataset_path.resolve()), "sha256": dataset_hash, "size_bytes": dataset_size, "kind": dataset_kind},
        "artifact_count": len(captured),
        "artifacts": captured,
        "manifest_semantics": "content-addressed provenance snapshot; retain this manifest separately from described artifacts",
        **FALSE_FLAGS,
    }
    return payload


def _assert_acyclic(entries: list[dict[str, Any]]) -> None:
    graph = {item["artifact_id"]: item["depends_on"] for item in entries}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"dependency_cycle:{node}")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def verify_provenance(
    manifest: dict[str, Any],
    *,
    root: Path,
    current_source_revision: str,
    config_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(manifest, dict) or manifest.get("audit_name") != "research_artifact_provenance_manifest":
        raise ValueError("provenance_manifest_invalid")
    revision = _validate_revision(current_source_revision)
    for key in FALSE_FLAGS:
        if manifest.get(key) is not False:
            errors.append({"code": "safety_flag_not_explicitly_false", "field": key})
    if manifest.get("source_revision") != revision:
        errors.append({"code": "STALE_CODE_REVISION", "expected": revision, "recorded": manifest.get("source_revision")})

    for label, path in (("config", config_path), ("dataset", dataset_path)):
        try:
            digest, size, kind = fingerprint_path(path)
            recorded = manifest.get(label)
            if not isinstance(recorded, dict):
                errors.append({"code": "MISSING_PROVENANCE", "field": label})
                continue
            if recorded.get("sha256") != digest or recorded.get("size_bytes") != size or recorded.get("kind") != kind:
                errors.append({"code": f"STALE_{label.upper()}", "field": label, "current_sha256": digest, "recorded_sha256": recorded.get("sha256")})
        except (OSError, ValueError) as exc:
            errors.append({"code": f"CURRENT_{label.upper()}_UNAVAILABLE", "detail": str(exc)})

    entries = manifest.get("artifacts")
    declared_artifact_count = manifest.get("artifact_count")
    if (
        not isinstance(entries, list)
        or not isinstance(declared_artifact_count, int)
        or isinstance(declared_artifact_count, bool)
        or declared_artifact_count != len(entries)
    ):
        errors.append({"code": "ARTIFACT_INVENTORY_INVALID"})
        entries = entries if isinstance(entries, list) else []
    root = root.resolve(strict=True)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_paths: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            errors.append({"code": "ARTIFACT_ENTRY_INVALID"})
            continue
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip() or artifact_id != artifact_id.strip() or artifact_id in seen:
            errors.append({"code": "ARTIFACT_ID_INVALID_OR_DUPLICATE", "artifact_id": artifact_id})
            continue
        seen.add(artifact_id)
        path_text = item.get("path")
        reasons: list[str] = []
        if item.get("source_revision") != revision:
            reasons.append("STALE_CODE_REVISION")
        if not SHA256_RE.fullmatch(item.get("sha256", "")):
            reasons.append("MISSING_OR_INVALID_ARTIFACT_HASH")
        path_is_safe = (
            isinstance(path_text, str)
            and bool(path_text)
            and path_text == path_text.strip()
            and "\\" not in path_text
            and not PurePosixPath(path_text).is_absolute()
            and ".." not in PurePosixPath(path_text).parts
            and PurePosixPath(path_text).as_posix() == path_text
            and path_text not in {".", ""}
        )
        if not path_is_safe:
            reasons.append("MISSING_OR_INVALID_ARTIFACT_PATH")
        elif path_text in seen_paths:
            reasons.append("DUPLICATE_ARTIFACT_PATH")
            errors.append({"code": "ARTIFACT_PATH_DUPLICATE", "artifact_id": artifact_id, "path": path_text})
        else:
            seen_paths.add(path_text)
            candidate = root / Path(path_text)
            try:
                if candidate.is_symlink():
                    raise ValueError(f"symlink_not_allowed:{candidate}")
                resolved = candidate.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    raise ValueError(f"artifact_path_outside_root:{candidate}")
                digest, size, kind = fingerprint_path(resolved)
                if digest != item.get("sha256") or size != item.get("size_bytes") or kind != item.get("kind"):
                    reasons.append("ARTIFACT_CONTENT_MISMATCH")
            except (OSError, ValueError) as exc:
                reasons.append("ARTIFACT_UNAVAILABLE_OR_UNSAFE")
                errors.append({"code": "ARTIFACT_UNAVAILABLE_OR_UNSAFE", "artifact_id": artifact_id, "detail": str(exc)})
        results.append({
            "artifact_id": artifact_id,
            "path": path_text,
            "status": "CURRENT" if not reasons else "STALE_OR_INVALID",
            "reasons": sorted(set(reasons)),
        })
    stale_count = sum(item["status"] != "CURRENT" for item in results)
    status = "PASS" if not errors and stale_count == 0 else "FAIL"
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_provenance_verification",
        "status": status,
        "artifact_count": len(results),
        "current_count": len(results) - stale_count,
        "stale_or_invalid_count": stale_count,
        "artifacts": sorted(results, key=lambda item: item["artifact_id"]),
        "errors": errors,
        **FALSE_FLAGS,
        "scope_limitations": [
            "A digest verifies byte consistency against this manifest, not semantic correctness or research validity.",
            "The manifest does not prove its own authenticity; retain it or its digest in a separately controlled location.",
            "The supplied source revision is checked for consistency but is not independently discovered from a Git checkout.",
            "Verification checks declared artifacts only; it cannot identify artifacts omitted from the manifest.",
            "This tool does not regenerate artifacts, execute experiments, train models, connect to brokers, trade, or authorize deployment.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--specification", required=True, type=Path)
    capture.add_argument("--source-revision", required=True)
    capture.add_argument("--config-path", required=True, type=Path)
    capture.add_argument("--dataset-path", required=True, type=Path)
    capture.add_argument("--root", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--current-source-revision", required=True)
    verify.add_argument("--config-path", required=True, type=Path)
    verify.add_argument("--dataset-path", required=True, type=Path)
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            report = capture_provenance(
                _read_object(args.specification),
                source_revision=args.source_revision,
                config_path=args.config_path,
                dataset_path=args.dataset_path,
                root=args.root,
            )
        else:
            report = verify_provenance(
                _read_object(args.manifest),
                root=args.root,
                current_source_revision=args.current_source_revision,
                config_path=args.config_path,
                dataset_path=args.dataset_path,
            )
        write_text_no_clobber(
            args.output,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps({
            "status": report.get("status", "CAPTURED"),
            "artifact_count": report["artifact_count"],
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report.get("status", "CAPTURED") in {"CAPTURED", "PASS"} else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

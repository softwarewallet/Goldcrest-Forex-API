"""Resolve Git provenance from a local checkout and require a clean source tree.

This utility reports the exact HEAD SHA and repository root from Git itself. By default, dirty
tracked or untracked files fail the provenance check so a mutable working tree is not mistaken
for a reproducible commit. Ignored files are not included by Git status.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
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


def _git(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"git_exit_{completed.returncode}"
        raise ValueError(f"git_command_failed:{args[0]}:{message}")
    return completed.stdout


def inspect_git_provenance(repo_path: Path) -> dict[str, Any]:
    root_text = _git(repo_path, "rev-parse", "--show-toplevel").strip()
    root = Path(root_text).resolve(strict=True)
    revision = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip().lower()
    if not REVISION_RE.fullmatch(revision):
        raise ValueError("git_head_is_not_full_40_character_commit_sha")
    branch = _git(root, "branch", "--show-current").strip() or None
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty_paths: list[str] = []
    if status:
        records = status.split("\0")
        for record in records:
            if not record:
                continue
            # Porcelain v1 uses two status columns followed by a path. Renames can include a
            # second NUL-separated path; record all paths without trying to infer rename meaning.
            dirty_paths.append(record[3:] if len(record) >= 3 else record)
    dirty_paths = sorted(set(dirty_paths))
    clean = not dirty_paths
    return {
        "schema_version": 1,
        "audit_name": "research_git_source_provenance",
        "status": "PASS" if clean else "FAIL",
        "repository_root": str(root),
        "source_revision": revision,
        "branch": branch,
        "working_tree_clean": clean,
        "dirty_path_count": len(dirty_paths),
        "dirty_paths": dirty_paths,
        "errors": [] if clean else [{"code": "WORKING_TREE_DIRTY", "dirty_path_count": len(dirty_paths)}],
        "ignored_files_included": False,
        **FALSE_FLAGS,
        "scope_limitations": [
            "The commit SHA identifies Git HEAD; dirty working-tree changes are not represented by that commit.",
            "Ignored files are not enumerated by Git status and should not contain source/config inputs used in a reproducibility run unless separately fingerprinted.",
            "This check does not fetch remote refs or establish that the commit is published to a trusted remote.",
            "A clean checkout does not establish experiment correctness, profitability, or deployment suitability.",
        ],
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"output_already_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = inspect_git_provenance(args.repo_path)
        _write_report(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "source_revision": report["source_revision"],
            "working_tree_clean": report["working_tree_clean"],
            "dirty_path_count": report["dirty_path_count"],
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

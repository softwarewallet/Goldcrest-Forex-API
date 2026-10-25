from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.research_git_source_provenance import inspect_git_provenance


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def init_repo(path: Path) -> None:
    path.mkdir()
    git(path, "init")
    git(path, "config", "user.email", "tests@example.com")
    git(path, "config", "user.name", "Research Tests")
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-m", "initial")


def test_clean_repository_returns_exact_full_head_sha(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    report = inspect_git_provenance(repo)
    assert report["status"] == "PASS"
    assert report["working_tree_clean"] is True
    assert report["source_revision"] == git(repo, "rev-parse", "HEAD")
    assert len(report["source_revision"]) == 40
    assert report["dirty_paths"] == []
    assert report["deployment_authorized"] is False


@pytest.mark.parametrize("change_kind", ["tracked", "untracked"])
def test_dirty_repository_fails_closed(tmp_path, change_kind):
    repo = tmp_path / "repo"
    init_repo(repo)
    if change_kind == "tracked":
        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    else:
        (repo / "new.txt").write_text("untracked\n", encoding="utf-8")
    report = inspect_git_provenance(repo)
    assert report["status"] == "FAIL"
    assert report["working_tree_clean"] is False
    assert report["dirty_path_count"] >= 1
    assert report["errors"][0]["code"] == "WORKING_TREE_DIRTY"


def test_ignored_files_are_not_reported(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "ignore generated temp")
    (repo / "ignored.tmp").write_text("ignored", encoding="utf-8")
    report = inspect_git_provenance(repo)
    assert report["status"] == "PASS"
    assert report["ignored_files_included"] is False


def test_non_repository_fails_with_clear_error(tmp_path):
    with pytest.raises(ValueError, match="git_command_failed"):
        inspect_git_provenance(tmp_path)

"""Atomic filesystem publication helpers for offline research audit outputs."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_no_clobber(path: Path, payload: str) -> None:
    """Atomically publish text without replacing an existing or concurrent target.

    The temporary file is created in the destination directory so hard-link
    publication stays on the same filesystem. Existing targets are never replaced.
    """
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise ValueError(f"output_already_exists:{path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

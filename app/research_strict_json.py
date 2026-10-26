"""Strict RFC 8259 JSON parsing helpers for offline research artifacts.

Python's default JSON decoder accepts duplicate object keys and non-standard numeric
constants. Integrity readers should reject those ambiguous inputs consistently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject JSON objects that repeat a key at the same object level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which RFC 8259 does not permit."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity at any depth."""
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )


def strict_json_load(path: str | Path) -> Any:
    """Read and strictly parse a UTF-8 JSON artifact."""
    source = Path(path)
    return strict_json_loads(source.read_text(encoding="utf-8"))


def strict_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON artifact and require an object root."""
    source = Path(path)
    value = strict_json_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{source.name} must contain a JSON object")
    return value

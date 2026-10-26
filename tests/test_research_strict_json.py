from __future__ import annotations

import pytest

from app.research_strict_json import strict_json_loads, strict_json_object


@pytest.mark.parametrize(
    "payload",
    [
        '{"outer": {"value": 1, "value": 2}}',
        '{"outer": [1, {"value": 1, "value": 2}]}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
    ],
)
def test_strict_json_rejects_duplicate_keys_and_nonstandard_constants(payload: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


def test_strict_json_accepts_standard_json_and_nested_values() -> None:
    assert strict_json_loads('{"items": [1, true, null, {"ok": "yes"}]}') == {
        "items": [1, True, None, {"ok": "yes"}]
    }


def test_strict_json_object_requires_object_root(tmp_path) -> None:
    path = tmp_path / "array.json"
    path.write_text('[1, 2, 3]', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        strict_json_object(path)


def test_strict_json_object_rejects_ambiguous_file(tmp_path) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_text('{"status": "PASS", "status": "FAIL"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_json_key:status"):
        strict_json_object(path)

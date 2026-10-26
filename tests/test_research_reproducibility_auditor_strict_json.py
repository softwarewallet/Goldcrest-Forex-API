from pathlib import Path

import pytest

from app.research_reproducibility_auditor import _read_object


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
    b'[]',
])
def test_reproducibility_reader_rejects_non_strict_json(tmp_path: Path, raw: bytes):
    path = tmp_path / "artifact.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError):
        _read_object(path)


def test_reproducibility_reader_accepts_valid_json_object(tmp_path: Path):
    path = tmp_path / "artifact.json"
    path.write_text('{"actionable":false,"nested":{"count":2}}', encoding="utf-8")

    assert _read_object(path) == {"actionable": False, "nested": {"count": 2}}

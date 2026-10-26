from __future__ import annotations

import pytest

import app.research_atomic_output as atomic_output


def test_write_text_no_clobber_publishes_payload_atomically(tmp_path):
    target = tmp_path / "nested" / "report.json"

    atomic_output.write_text_no_clobber(target, '{"status":"PASS"}\n')

    assert target.read_text(encoding="utf-8") == '{"status":"PASS"}\n'
    assert list(target.parent.iterdir()) == [target]


def test_write_text_no_clobber_rejects_existing_target_without_modifying_it(tmp_path):
    target = tmp_path / "report.json"
    original = '{"owner":"existing"}\n'
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_output.write_text_no_clobber(target, '{"status":"PASS"}\n')

    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [target]


def test_write_text_no_clobber_does_not_overwrite_concurrent_target(monkeypatch, tmp_path):
    target = tmp_path / "report.json"
    competitor_payload = '{"owner":"concurrent-writer"}\n'

    def competitor_writes_before_link(source, destination):
        from pathlib import Path

        Path(destination).write_text(competitor_payload, encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(atomic_output.os, "link", competitor_writes_before_link)

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_output.write_text_no_clobber(target, '{"status":"PASS"}\n')

    assert target.read_text(encoding="utf-8") == competitor_payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]



def test_write_text_atomic_replace_updates_existing_payload(tmp_path):
    target = tmp_path / "receipt.json"
    target.write_text('{"version":1}\\n', encoding="utf-8")

    atomic_output.write_text_atomic_replace(target, '{"version":2}\\n')

    assert target.read_text(encoding="utf-8") == '{"version":2}\\n'
    assert list(tmp_path.iterdir()) == [target]

from __future__ import annotations

import json

import pytest

from app.research_label_eligibility_threshold_sweep import (
    choose_threshold_report_path,
    parse_elapsed_caps,
)


def test_parse_elapsed_caps_preserves_order():
    assert parse_elapsed_caps("1,1.5,8") == [1.0, 1.5, 8.0]


@pytest.mark.parametrize(
    "value, message",
    [
        ("", "at least two"),
        ("1", "at least two"),
        ("1,,2", "comma-separated"),
        ("1,1", "unique"),
        ("0,1", "finite and positive"),
        ("-1,2", "finite and positive"),
        ("nan,2", "finite and positive"),
        ("inf,2", "finite and positive"),
        ("one,2", "comma-separated"),
        ([1.0, True], "finite and positive"),
    ],
)
def test_parse_elapsed_caps_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        parse_elapsed_caps(value)


def test_threshold_report_path_auto_versions_without_overwriting(tmp_path):
    first = choose_threshold_report_path(tmp_path, "eurusd-cap-sweep")
    first.write_text('{"preserve": true}\n', encoding="utf-8")
    second = choose_threshold_report_path(tmp_path, "eurusd-cap-sweep")
    assert first.name == "eurusd-cap-sweep_threshold_sweep_report.json"
    assert second.name == "eurusd-cap-sweep-rerun-2_threshold_sweep_report.json"
    assert json.loads(first.read_text(encoding="utf-8")) == {"preserve": True}


def test_threshold_report_prefix_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="safe filename"):
        choose_threshold_report_path(tmp_path, "../escape")

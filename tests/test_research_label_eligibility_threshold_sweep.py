from __future__ import annotations

import json

import pytest

from app.research_label_eligibility_threshold_sweep import (
    _read_json,
    _fold_signature,
    _fold_window_signature,
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


def _sweep_report_with_eligible_rows(eligible_rows: int) -> dict:
    return {
        "seed_results": [
            {
                "experiments": {"xgboost/all_features": {}},
                "fold_signature": [
                    [0, "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", eligible_rows],
                    [1, "2026-01-07T00:00:00Z", "2026-01-14T00:00:00Z", eligible_rows - 2],
                ],
            }
        ]
    }


def test_cross_cap_fold_window_signature_ignores_cap_dependent_eligible_counts():
    one_hour = _sweep_report_with_eligible_rows(100)
    one_and_half_hours = _sweep_report_with_eligible_rows(106)

    # Per-cap seed consistency keeps eligible counts; only cross-cap window equality drops them.
    assert _fold_signature(one_hour) != _fold_signature(one_and_half_hours)
    assert _fold_window_signature(one_hour) == _fold_window_signature(one_and_half_hours)
    assert _fold_window_signature(one_hour) == [
        [0, "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"],
        [1, "2026-01-07T00:00:00Z", "2026-01-14T00:00:00Z"],
    ]


@pytest.mark.parametrize(
    "payload",
    [
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
    ],
)
def test_json_reader_rejects_ambiguous_or_nonstandard_json(tmp_path, payload):
    path = tmp_path / "malformed.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _read_json(path)

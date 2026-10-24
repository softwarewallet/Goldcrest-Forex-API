from __future__ import annotations

from pathlib import Path

import pytest

from app.research_label_eligibility_paired_replay import choose_paired_replay_run_ids


def test_pair_run_ids_are_distinct_and_initially_unversioned(tmp_path):
    uncapped, capped, versioned = choose_paired_replay_run_ids(
        tmp_path,
        "baseline-paired-cap-1h",
        symbol="EURUSD",
        timeframe="M15",
    )

    assert uncapped == "baseline-paired-cap-1h-uncapped"
    assert capped == "baseline-paired-cap-1h-capped"
    assert uncapped != capped
    assert versioned is False


def test_pair_run_ids_auto_version_together_when_either_namespace_is_occupied(tmp_path):
    (tmp_path / "baseline-paired-cap-1h-capped_pipeline_manifest.json").write_text(
        '{"preserve": true}\n',
        encoding="utf-8",
    )

    uncapped, capped, versioned = choose_paired_replay_run_ids(
        tmp_path,
        "baseline-paired-cap-1h",
        symbol="EURUSD",
        timeframe="M15",
    )

    assert uncapped == "baseline-paired-cap-1h-uncapped-rerun-2"
    assert capped == "baseline-paired-cap-1h-capped-rerun-2"
    assert versioned is True
    assert (tmp_path / "baseline-paired-cap-1h-capped_pipeline_manifest.json").read_text(
        encoding="utf-8"
    ) == '{"preserve": true}\n'


def test_pair_run_ids_respect_custom_receipt_collisions(tmp_path):
    (tmp_path / "custom-capped_paired_replay_receipt.json").write_text(
        '{"keep": true}\n',
        encoding="utf-8",
    )
    uncapped, capped, versioned = choose_paired_replay_run_ids(
        tmp_path,
        "custom",
        symbol="EURUSD",
        timeframe="M15",
    )
    assert uncapped == "custom-uncapped-rerun-2"
    assert capped == "custom-capped-rerun-2"
    assert versioned is True


@pytest.mark.parametrize("prefix", ["", "../escape", "bad/name", "a" * 91, ".."])
def test_pair_run_ids_reject_unsafe_prefix(prefix, tmp_path):
    with pytest.raises(ValueError, match="run_id_prefix"):
        choose_paired_replay_run_ids(
            tmp_path,
            prefix,
            symbol="EURUSD",
            timeframe="M15",
        )


def test_pair_run_ids_reject_overlong_final_ids(tmp_path):
    with pytest.raises(ValueError, match="128-character"):
        choose_paired_replay_run_ids(
            tmp_path,
            "p" * 89,
            symbol="EURUSD",
            timeframe="M15",
        )

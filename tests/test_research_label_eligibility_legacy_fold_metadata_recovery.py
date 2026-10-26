from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_label_eligibility_legacy_fold_metadata_recovery import (
    _read_json,
    recover_seed_report,
    recover_threshold_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _comparison(counts: list[int], *, start: str = "2026-01-01T00:00:00Z") -> dict:
    folds = [
        {
            "fold_index": i,
            "test_start_utc": start,
            "test_end_utc": "2026-01-02T00:00:00Z",
            "eligible_test_rows": count,
        }
        for i, count in enumerate(counts)
    ]
    return {
        "status": "PASS_WITH_WARNINGS",
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "inferential": False,
        "raw_ohlc_sha256": "a" * 64,
        "experiments": [
            {
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "fold_results": folds,
            }
        ],
    }


def _seed_report(tmp_path: Path, counts_by_seed: list[list[int]]) -> Path:
    seeds = list(range(20261021, 20261021 + len(counts_by_seed)))
    records = []
    for seed, counts in zip(seeds, counts_by_seed, strict=True):
        comparison_path = _write(
            tmp_path / f"comparison-{seed}.json",
            _comparison(counts),
        )
        records.append({
            "seed": seed,
            "status": "PASS_WITH_WARNINGS",
            "comparison_report_path": str(comparison_path),
            "raw_ohlc_sha256": "a" * 64,
        })
    return _write(tmp_path / "seed-sweep.json", {
        "status": "PASS_WITH_WARNINGS",
        "max_label_elapsed_hours": 1.5,
        "requested_seeds": seeds,
        "successful_seed_count": len(seeds),
        "failed_seed_count": 0,
        "raw_ohlc_sha256": "a" * 64,
        "seed_results": records,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "inferential": False,
    })


def test_recovery_adds_fold_counts_and_preserves_source(tmp_path):
    source = _seed_report(tmp_path, [[100, 101], [100, 101]])
    original_bytes = source.read_bytes()
    output = tmp_path / "recovered.json"

    recovered = recover_seed_report(source, 1.5, output)

    assert [r["eligible_test_rows"] for r in recovered["fold_eligibility_counts"]] == [100, 101]
    assert all("fold_eligibility_counts" in row for row in recovered["seed_results"])
    assert recovered["fold_eligibility_metadata_recovery"]["model_retraining_performed"] is False
    assert source.read_bytes() == original_bytes
    assert json.loads(output.read_text(encoding="utf-8"))["fold_eligibility_counts"] == recovered["fold_eligibility_counts"]


def test_recovery_rejects_seed_count_mismatch(tmp_path):
    source = _seed_report(tmp_path, [[100, 101], [100, 102]])

    with pytest.raises(ValueError, match="fold_counts_differ_between_seeds"):
        recover_seed_report(source, 1.5, tmp_path / "recovered.json")


def test_recovery_never_overwrites_existing_output(tmp_path):
    source = _seed_report(tmp_path, [[100, 101], [100, 101]])
    output = tmp_path / "recovered.json"
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        recover_seed_report(source, 1.5, output)

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_recovery_rejects_unsafe_comparison_flags(tmp_path):
    source = _seed_report(tmp_path, [[100, 101], [100, 101]])
    payload = json.loads(source.read_text(encoding="utf-8"))
    comparison_path = Path(payload["seed_results"][0]["comparison_report_path"])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["execution_connected"] = True
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected_safety_flag:execution_connected"):
        recover_seed_report(source, 1.5, tmp_path / "recovered.json")



def test_threshold_recovery_updates_only_matching_caps_and_preserves_research_safety(tmp_path):
    digest = "a" * 64
    rows_by_cap = {
        1.0: [100, 101],
        1.5: [120, 121],
        8.0: [120, 121],
    }
    recovered_reports = {}
    for cap, counts in rows_by_cap.items():
        rows = [
            {
                "fold_index": i,
                "test_start_utc": f"2026-01-0{i + 1}T00:00:00Z",
                "test_end_utc": f"2026-01-0{i + 2}T00:00:00Z",
                "eligible_test_rows": count,
            }
            for i, count in enumerate(counts)
        ]
        seed_report = {
            "raw_ohlc_sha256": digest,
            "successful_seed_count": 2,
            "fold_eligibility_counts": rows,
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
            "inferential": False,
        }
        recovered_reports[cap] = (seed_report, tmp_path / f"recovered-{cap}.json")

    source = _write(tmp_path / "threshold-source.json", {
        "status": "PASS_WITH_WARNINGS",
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP",
        "raw_ohlc_sha256": digest,
        "requested_caps_hours": [1.0, 1.5, 8.0],
        "cap_results": [
            {"max_label_elapsed_hours": cap, "status": "PASS_WITH_WARNINGS"}
            for cap in (1.0, 1.5, 8.0)
        ],
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "inferential": False,
    })
    output = tmp_path / "threshold-recovered.json"

    recovered = recover_threshold_report(source, output, recovered_reports)

    assert [row["max_label_elapsed_hours"] for row in recovered["cap_results"]] == [1.0, 1.5, 8.0]
    assert [
        row["eligible_test_rows"] for row in recovered["cap_results"][1]["fold_eligibility_counts"]
    ] == [120, 121]
    assert recovered["actionable"] is False
    assert recovered["execution_connected"] is False
    assert recovered["automatic_model_selection"] is False
    assert recovered["inferential"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["fold_eligibility_metadata_recovery"]["source_artifacts_modified"] is False


def test_threshold_recovery_rejects_hash_mismatch(tmp_path):
    source = _write(tmp_path / "threshold-source.json", {
        "status": "PASS",
        "raw_ohlc_sha256": "b" * 64,
        "cap_results": [
            {"max_label_elapsed_hours": cap, "status": "PASS"}
            for cap in (1.0, 1.5, 8.0)
        ],
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "inferential": False,
    })
    recovered_reports = {
        cap: ({
            "raw_ohlc_sha256": "a" * 64,
            "successful_seed_count": 2,
            "fold_eligibility_counts": [{
                "fold_index": 0,
                "test_start_utc": "2026-01-01T00:00:00Z",
                "test_end_utc": "2026-01-02T00:00:00Z",
                "eligible_test_rows": 100,
            }],
        }, tmp_path / f"recovered-{cap}.json")
        for cap in (1.0, 1.5, 8.0)
    }

    from app.research_label_eligibility_legacy_fold_metadata_recovery import recover_threshold_report

    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        recover_threshold_report(source, tmp_path / "out.json", recovered_reports)


@pytest.mark.parametrize(
    "payload",
    [
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
    ],
)
def test_read_json_rejects_ambiguous_or_nonstandard_json(tmp_path, payload):
    path = tmp_path / "malformed.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _read_json(path)

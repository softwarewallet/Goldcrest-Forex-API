from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.research_label_eligibility_variant import _infer_fold_geometry


def _write_dataset_timestamps(path: Path, count: int) -> list[str]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        (start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z")
        for index in range(count)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"timestamp": timestamp} for timestamp in timestamps)
    return timestamps


def _matrix_for_geometry(timestamps: list[str]) -> dict:
    horizon = 4
    initial_train = 80
    validation = 30
    test = 40
    step = 20
    windows = []
    fold_results = []
    train_end = initial_train
    while train_end + validation + test <= len(timestamps):
        validation_end = train_end + validation
        test_start = validation_end + horizon
        test_end = validation_end + test - 1
        window = {
            "fold_index": len(windows),
            "test_start_utc": timestamps[test_start],
            "test_end_utc": timestamps[test_end],
            "test_rows": test - horizon,
        }
        windows.append(window)
        fold_results.append({
            "fold_index": len(fold_results),
            "validation_start_utc": timestamps[train_end + horizon],
            "test_start_utc": timestamps[test_start],
            "test_end_utc": timestamps[test_end],
            "trials_requested": 5,
        })
        train_end += step
    return {
        "common_fold_test_windows": windows,
        "experiments": [
            {"fold_results": fold_results},
            {"fold_results": [dict(fold) for fold in fold_results]},
        ],
    }


def test_infer_fold_geometry_recovers_exact_pre_purge_sizes_and_checks_all_windows(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)

    recovered = _infer_fold_geometry(dataset, matrix, horizon_bars=4)

    assert recovered == {
        "initial_train_bars": 80,
        "validation_bars": 30,
        "test_bars": 40,
        "step_bars": 20,
        "fold_count": 5,
        "dataset_rows": 230,
    }


def test_infer_fold_geometry_rejects_test_window_mismatch(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["common_fold_test_windows"][2]["test_end_utc"] = timestamps[200]

    with pytest.raises(ValueError, match="reconstructed_test_window_mismatch:fold_2"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)


def test_infer_fold_geometry_rejects_timestamps_missing_from_dataset(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["experiments"][0]["fold_results"][0]["validation_start_utc"] = "2030-01-01T00:00:00Z"

    with pytest.raises(ValueError, match="fold_timestamp_not_in_dataset"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)


def test_infer_fold_geometry_requires_multiple_windows_when_geometry_is_not_persisted(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["common_fold_test_windows"] = matrix["common_fold_test_windows"][:1]
    matrix["experiments"] = [
        {"fold_results": experiment["fold_results"][:1]}
        for experiment in matrix["experiments"]
    ]

    with pytest.raises(ValueError, match="without_two_common_test_windows"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)

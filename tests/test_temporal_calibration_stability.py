import pytest

from app.temporal_calibration_stability import build_temporal_calibration_stability_audit


def fixture():
    folds = {"model": []}
    boundaries = []
    for i, (ece, low, high) in enumerate([(0.1, 0.08, 0.12), (0.16, 0.13, 0.18), (0.3, 0.25, 0.34)]):
        folds["model"].append({
            "fold_index": i,
            "validation": {"expected_calibration_error": ece, "sample_count": 40,
                           "calibration_uncertainty": {"interval_low": low, "interval_high": high}},
            "test": {"expected_calibration_error": ece, "sample_count": 40,
                     "calibration_uncertainty": {"interval_low": low, "interval_high": high}},
        })
        boundaries.append({"fold_index": i, "test_start_utc": f"2026-0{i+1}-01T00:00:00Z",
                           "test_end_utc": f"2026-0{i+1}-20T00:00:00Z"})
    return folds, boundaries


def test_temporal_calibration_audit_sorts_by_explicit_boundaries_and_flags_endpoint_separation():
    folds, boundaries = fixture()
    folds["model"] = list(reversed(folds["model"]))
    audit = build_temporal_calibration_stability_audit(folds, boundaries)
    test_row = next(row for row in audit["comparisons"] if row["partition"] == "test")
    assert test_row["fold_indices"] == [0, 1, 2]
    assert test_row["first_to_last_ece_delta"] == pytest.approx(0.2)
    assert test_row["status"] == "ENDPOINT_ECE_INCREASE_INTERVALS_DISJOINT"
    assert test_row["actionable"] is False


def test_temporal_audit_withholds_interpretation_without_three_periods():
    folds, boundaries = fixture()
    audit = build_temporal_calibration_stability_audit(
        {"model": folds["model"][:2]}, boundaries[:2]
    )
    assert all(row["status"] == "INSUFFICIENT_TEMPORAL_COVERAGE" for row in audit["comparisons"])


def test_temporal_audit_does_not_infer_order_without_declared_boundaries():
    folds, _ = fixture()
    audit = build_temporal_calibration_stability_audit(folds, [])
    assert all(row["period_count"] == 0 for row in audit["comparisons"])
    assert all(row["status"] == "INSUFFICIENT_TEMPORAL_COVERAGE" for row in audit["comparisons"])


def test_temporal_audit_rejects_duplicate_or_invalid_boundaries():
    folds, boundaries = fixture()
    with pytest.raises(ValueError, match="duplicate fold index"):
        build_temporal_calibration_stability_audit(folds, boundaries + [boundaries[0]])
    broken = [dict(boundaries[0], test_start_utc="2026-01-30T00:00:00Z")]
    with pytest.raises(ValueError, match="start before end"):
        build_temporal_calibration_stability_audit(folds, broken)


def _matrix_fixture():
    import hashlib
    import json

    windows = []
    folds = []
    ece_values = [(0.10, 0.08, 0.12), (0.16, 0.13, 0.18), (0.30, 0.25, 0.34)]
    for index, (ece, low, high) in enumerate(ece_values):
        start = f"2026-0{index + 1}-01T00:00:00Z"
        end = f"2026-0{index + 1}-20T00:00:00Z"
        windows.append({
            "fold_index": index,
            "test_start_utc": start,
            "test_end_utc": end,
            "test_rows": 40,
        })
        folds.append({
            "fold_index": index,
            "test_start_utc": start,
            "test_end_utc": end,
            "test_metrics": {
                "sample_count": 40,
                "expected_calibration_error": ece,
                "calibration_uncertainty": {"interval_low": low, "interval_high": high},
            },
        })
    payload = {
        "schema_version": 1,
        "run_id": "temporal-calibration-cli-test",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "horizon_bars": 4,
        "fold_count": 3,
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "report_warning": "Offline research only.",
        "common_fold_test_windows": windows,
        "experiments": [{
            "experiment_id": "xgboost-all_features",
            "actionable": False,
            "test_metrics_used_for_tuning": False,
            "fold_results": folds,
        }],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _rehash_matrix_fixture(matrix):
    import hashlib
    import json

    matrix.pop("report_payload_sha256", None)
    matrix["report_payload_sha256"] = hashlib.sha256(
        json.dumps(matrix, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return matrix


def test_report_rejects_boolean_schema_version(tmp_path):
    from app.temporal_calibration_stability import build_temporal_calibration_stability_report

    matrix = _rehash_matrix_fixture(_matrix_fixture())
    matrix["schema_version"] = True
    matrix = _rehash_matrix_fixture(matrix)
    matrix_path = tmp_path / "boolean-schema-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    report = build_temporal_calibration_stability_report(matrix, matrix_path=matrix_path)

    assert report["status"] == "FAIL"
    assert "unsupported_or_missing_schema_version" in report["errors"]


def test_report_warns_when_sample_count_is_invalid(tmp_path):
    from app.temporal_calibration_stability import build_temporal_calibration_stability_report

    matrix = _matrix_fixture()
    matrix["experiments"][0]["fold_results"][0]["test_metrics"]["sample_count"] = 0
    matrix = _rehash_matrix_fixture(matrix)
    matrix_path = tmp_path / "invalid-sample-count-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    report = build_temporal_calibration_stability_report(matrix, matrix_path=matrix_path)

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "sample_count_missing_or_invalid:xgboost-all_features:fold_0" in report["warnings"]


def test_cli_help_is_available(capsys):
    from app.temporal_calibration_stability import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "--experiment-matrix" in capsys.readouterr().out


def test_cli_builds_report_from_experiment_matrix(tmp_path, capsys):
    import json
    from app.temporal_calibration_stability import main

    matrix_path = tmp_path / "matrix.json"
    report_path = tmp_path / "temporal_calibration.json"
    matrix_path.write_text(json.dumps(_matrix_fixture()), encoding="utf-8")

    exit_code = main([
        "--experiment-matrix", str(matrix_path),
        "--output", str(report_path),
    ])
    output = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output["status"] == "PASS"
    assert output["experiment_matrix_payload_hash_verified"] is True
    assert output["comparison_count"] == 1
    row = output["comparisons"][0]
    assert row["partition"] == "test"
    assert row["fold_indices"] == [0, 1, 2]
    assert row["status"] == "ENDPOINT_ECE_INCREASE_INTERVALS_DISJOINT"
    assert output["actionable"] is False
    assert output["execution_connected"] is False
    assert "Report written to:" in capsys.readouterr().err


def test_cli_refuses_to_overwrite_existing_report(tmp_path):
    import json
    from app.temporal_calibration_stability import main

    matrix_path = tmp_path / "matrix.json"
    report_path = tmp_path / "already_exists.json"
    matrix_path.write_text(json.dumps(_matrix_fixture()), encoding="utf-8")
    report_path.write_text("preserve-me", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([
            "--experiment-matrix", str(matrix_path),
            "--output", str(report_path),
        ])
    assert error.value.code == 2
    assert report_path.read_text(encoding="utf-8") == "preserve-me"


def test_report_rejects_experiment_missing_a_common_temporal_fold(tmp_path):
    import hashlib
    import json

    from app.temporal_calibration_stability import build_temporal_calibration_stability_report

    matrix = _matrix_fixture()
    matrix["experiments"][0]["fold_results"].pop()
    matrix.pop("report_payload_sha256", None)
    matrix["report_payload_sha256"] = hashlib.sha256(
        json.dumps(matrix, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    matrix_path = tmp_path / "incomplete-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    report = build_temporal_calibration_stability_report(matrix, matrix_path=matrix_path)

    assert report["status"] == "FAIL"
    assert any("fold_coverage_mismatch" in error for error in report["errors"])


def test_experiment_matrix_writer_and_temporal_audit_round_trip_is_reproducible(tmp_path):
    import json

    from app.research_experiment_matrix import (
        ExperimentMatrixReport,
        write_experiment_matrix_report,
    )
    from app.temporal_calibration_stability import write_temporal_calibration_stability_report

    windows = []
    folds = []
    for index, (ece, low, high) in enumerate([
        (0.10, 0.08, 0.12),
        (0.16, 0.13, 0.18),
        (0.30, 0.25, 0.34),
    ]):
        start = f"2026-0{index + 1}-01T00:00:00Z"
        end = f"2026-0{index + 1}-20T00:00:00Z"
        windows.append({
            "fold_index": index,
            "test_start_utc": start,
            "test_end_utc": end,
            "test_rows": 40,
        })
        folds.append({
            "fold_index": index,
            "train_start_utc": "2025-01-01T00:00:00Z",
            "train_end_utc": "2025-09-01T00:00:00Z",
            "validation_start_utc": "2025-09-02T00:00:00Z",
            "validation_end_utc": "2025-12-01T00:00:00Z",
            "test_start_utc": start,
            "test_end_utc": end,
            "model_type": "xgboost",
            "best_params": {"max_depth": 3},
            "validation_log_loss": 0.55,
            "trials_requested": 2,
            "trials_completed": 2,
            "train_rows_available_for_tuning": 100,
            "train_rows_excluded_unmatured": 0,
            "refit_rows_available": 120,
            "refit_rows_excluded_unmatured": 0,
            "test_metrics": {
                "sample_count": 40,
                "positive_count": 20,
                "negative_count": 20,
                "prevalence": 0.5,
                "brier_score": 0.22,
                "log_loss": 0.62,
                "accuracy_at_fixed_half": 0.55,
                "mean_predicted_probability": 0.51,
                "expected_calibration_error": ece,
                "reliability_bins": [],
                "observation_identity_sha256": "b" * 64,
                "calibration_uncertainty": {
                    "interval_low": low,
                    "interval_high": high,
                    "sample_count": 40,
                    "bins": 10,
                    "block_length": 4,
                    "bootstrap_resamples": 500,
                    "confidence_level": 0.95,
                    "status": "DESCRIPTIVE_MOVING_BLOCK_PERCENTILE_INTERVAL",
                },
            },
            "test_predictions": [],
            "baseline_diagnostics": {"baseline_labels_used_test_data": False},
            "actionable": False,
            "train_rows_excluded_ineligible_for_tuning": 0,
            "validation_rows_excluded_ineligible": 0,
            "refit_rows_excluded_ineligible": 0,
            "test_rows_available_before_eligibility": 40,
            "test_rows_excluded_ineligible": 0,
        })

    matrix_report = ExperimentMatrixReport(
        schema_version=1,
        run_id="temporal-matrix-writer-roundtrip",
        dataset_sha256="a" * 64,
        symbol="EURUSD",
        timeframe="M15",
        feature_names=("ret_1",),
        model_types=("xgboost",),
        fold_count=3,
        experiments=({
            "experiment_id": "temporal-matrix-writer-roundtrip-xgboost-all_features",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "feature_names": ["ret_1"],
            "feature_count": 1,
            "feature_schema_sha256": "c" * 64,
            "fold_count": 3,
            "fold_results": folds,
            "test_metrics_used_for_tuning": False,
            "used_common_fold_plan": True,
            "actionable": False,
        },),
        common_fold_test_windows=tuple(windows),
        horizon_bars=4,
        class_1_means="UP",
    )
    matrix_path = tmp_path / "temporal-matrix.json"
    write_experiment_matrix_report(matrix_report, matrix_path)

    first_path = tmp_path / "temporal-calibration-first.json"
    second_path = tmp_path / "temporal-calibration-second.json"
    first = write_temporal_calibration_stability_report(matrix_path, first_path)
    second = write_temporal_calibration_stability_report(matrix_path, second_path)

    assert first["status"] == "PASS"
    assert first["experiment_matrix_payload_hash_verified"] is True
    assert first["experiment_matrix_sha256"] == __import__("hashlib").sha256(matrix_path.read_bytes()).hexdigest()
    assert first["comparison_count"] == 1
    row = first["comparisons"][0]
    assert row["fold_indices"] == [0, 1, 2]
    assert row["first_to_last_ece_delta"] == pytest.approx(0.20)
    assert row["status"] == "ENDPOINT_ECE_INCREASE_INTERVALS_DISJOINT"
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()

    with pytest.raises(FileExistsError):
        write_temporal_calibration_stability_report(matrix_path, first_path)


def test_temporal_audit_normalizes_boundary_offsets_to_utc_before_ordering():
    folds, boundaries = fixture()
    boundaries[0]["test_start_utc"] = "2026-01-01T01:00:00+01:00"
    boundaries[0]["test_end_utc"] = "2026-01-20T01:00:00+01:00"

    audit = build_temporal_calibration_stability_audit(folds, boundaries)
    periods = next(row for row in audit["comparisons"] if row["partition"] == "test")["periods"]

    assert periods[0]["test_start_utc"] == "2026-01-01T00:00:00Z"
    assert periods[0]["test_end_utc"] == "2026-01-20T00:00:00Z"


def test_temporal_audit_rejects_invalid_partitions():
    folds, boundaries = fixture()
    with pytest.raises(ValueError, match="partitions must contain unique values"):
        build_temporal_calibration_stability_audit(folds, boundaries, partitions=({},))

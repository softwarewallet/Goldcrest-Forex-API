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


def test_temporal_audit_rejects_invalid_partitions():
    folds, boundaries = fixture()
    with pytest.raises(ValueError, match="partitions must contain unique values"):
        build_temporal_calibration_stability_audit(folds, boundaries, partitions=({},))

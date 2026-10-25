from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import ctrader_history_ingest
from app.research_provenance_auditor import audit_experiment_matrix_file
from app.research_validation import collect_research_preflight, summarize_experiment_matrix, validate_completed_run, validate_prepared_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_dataset(path: Path, rows: int = 10) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
            "label", "label_available_at", "return_1",
        ])
        writer.writeheader()
        for index in range(rows):
            ts = start + timedelta(minutes=15 * index)
            available = ts + timedelta(minutes=30)
            close = 1.1 + index * 0.0001
            writer.writerow({
                "symbol": "EURUSD",
                "timeframe": "M15",
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 0.00005,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
                "label": str(index % 2),
                "label_available_at": available.isoformat().replace("+00:00", "Z"),
                "return_1": index / 1000,
            })


def _build_complete_run(tmp_path: Path) -> Path:
    raw = tmp_path / "raw_ohlc.csv"
    raw.write_text(
        "timestamp,open,high,low,close\n"
        "2026-01-01T00:00:00Z,1.1,1.2,1.0,1.15\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "EURUSD_M15_research_dataset.csv"
    _write_dataset(dataset)
    dataset_manifest = tmp_path / "dataset.manifest.json"
    _write_json(dataset_manifest, {"output_dataset_sha256": _sha(dataset)})

    windows = [
        {"fold_index": 0, "test_start_utc": "2026-01-01T01:00:00Z", "test_end_utc": "2026-01-01T01:45:00Z", "test_rows": 4},
        {"fold_index": 1, "test_start_utc": "2026-01-01T01:30:00Z", "test_end_utc": "2026-01-01T02:15:00Z", "test_rows": 4},
    ]
    fold_results = [
        {
            "fold_index": 0,
            "train_start_utc": "2026-01-01T00:00:00Z",
            "train_end_utc": "2026-01-01T00:15:00Z",
            "validation_start_utc": "2026-01-01T00:30:00Z",
            "validation_end_utc": "2026-01-01T00:45:00Z",
            "test_start_utc": windows[0]["test_start_utc"],
            "test_end_utc": windows[0]["test_end_utc"],
            "best_params": {"max_depth": 3},
            "trials_requested": 2,
            "trials_completed": 2,
            "test_metrics_used_for_tuning": False,
            "test_metrics": {
                "sample_count": 4, "positive_count": 2, "negative_count": 2,
                "prevalence": 0.5, "brier_score": 0.2, "log_loss": 0.6,
                "accuracy_at_fixed_half": 0.6, "mean_predicted_probability": 0.51,
                "expected_calibration_error": 0.08,
            },
        },
        {
            "fold_index": 1,
            "train_start_utc": "2026-01-01T00:00:00Z",
            "train_end_utc": "2026-01-01T00:30:00Z",
            "validation_start_utc": "2026-01-01T00:45:00Z",
            "validation_end_utc": "2026-01-01T01:00:00Z",
            "test_start_utc": windows[1]["test_start_utc"],
            "test_end_utc": windows[1]["test_end_utc"],
            "best_params": {"max_depth": 4},
            "trials_requested": 2,
            "trials_completed": 2,
            "test_metrics_used_for_tuning": False,
            "test_metrics": {
                "sample_count": 4, "positive_count": 2, "negative_count": 2,
                "prevalence": 0.4, "brier_score": 0.22, "log_loss": 0.62,
                "accuracy_at_fixed_half": 0.55, "mean_predicted_probability": 0.49,
                "expected_calibration_error": 0.09,
            },
        },
    ]
    matrix = {
        "schema_version": 1,
        "run_id": "validation-fixture-01",
        "dataset_sha256": _sha(dataset),
        "symbol": "EURUSD",
        "timeframe": "M15",
        "model_types": ["xgboost"],
        "fold_count": 2,
        "feature_names": ["return_1"],
        "common_fold_test_windows": windows,
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "experiments": [{
            "experiment_id": "validation-fixture-01-xgboost-all",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "feature_names": ["return_1"],
            "feature_schema_sha256": "b" * 64,
            "fold_count": 2,
            "test_metrics_used_for_tuning": False,
            "actionable": False,
            "fold_results": fold_results,
        }],
    }
    report_path = tmp_path / "validation-fixture-01_experiment_matrix.json"
    _write_json(report_path, matrix)
    audit_path = tmp_path / "validation-fixture-01_provenance_audit.json"
    provenance = audit_experiment_matrix_file(report_path, audit_path)
    assert provenance["status"] == "PASS"

    manifest = {
        "schema_version": 1,
        "run_id": "validation-fixture-01",
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "label_horizon_bars": 2,
        "raw_ohlc_path": str(raw.resolve()),
        "raw_ohlc_sha256": _sha(raw),
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": _sha(dataset),
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": _sha(dataset_manifest),
        "experiment_matrix_report": str(report_path.resolve()),
        "experiment_matrix_report_sha256": _sha(report_path),
        "provenance_audit_path": str(audit_path.resolve()),
        "provenance_audit_sha256": _sha(audit_path),
        "provenance_audit_status": provenance["status"],
        "strict_provenance_audit": False,
        "ingestion": {"actionable": False},
        "class_1_means": "UP",
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "execution_connected": False,
        "warning": "Offline research only.",
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "validation-fixture-01_pipeline_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_prepared_dataset_validation_accepts_chronological_finite_data(tmp_path):
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset, rows=8)

    result = validate_prepared_dataset(dataset, expected_symbol="EURUSD", expected_timeframe="M15")

    assert result["status"] == "PASS"
    assert result["row_count"] == 8
    assert result["feature_names"] == ["return_1"]
    assert result["label_distribution"]["positive_count"] == 4
    assert result["actionable"] is False


def test_prepared_dataset_rejects_label_available_at_feature_timestamp(tmp_path):
    dataset = tmp_path / "zero_maturity.csv"
    _write_dataset(dataset, rows=4)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1]["label_available_at"] = rows[1]["timestamp"]
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\\n")
        writer.writeheader()
        writer.writerows(rows)

    result = validate_prepared_dataset(dataset)

    assert result["status"] == "FAIL"
    assert any("label_available_not_after_feature" in error for error in result["errors"])


def test_prepared_dataset_rejects_nonchronological_label_maturity_times(tmp_path):
    dataset = tmp_path / "maturity_order.csv"
    _write_dataset(dataset, rows=5)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows[0]["label_available_at"] = (start + timedelta(minutes=90)).isoformat().replace("+00:00", "Z")
    rows[1]["label_available_at"] = (start + timedelta(minutes=45)).isoformat().replace("+00:00", "Z")
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\\n")
        writer.writeheader()
        writer.writerows(rows)

    result = validate_prepared_dataset(dataset)

    assert result["status"] == "FAIL"
    assert any("label_available_timestamps_not_strictly_chronological" in error for error in result["errors"])


def test_prepared_dataset_rejects_duplicate_label_maturity_times(tmp_path):
    dataset = tmp_path / "duplicate_maturity.csv"
    _write_dataset(dataset, rows=4)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1]["label_available_at"] = rows[0]["label_available_at"]
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\\n")
        writer.writeheader()
        writer.writerows(rows)

    result = validate_prepared_dataset(dataset)

    assert result["status"] == "FAIL"
    assert any("duplicate_label_available_timestamp" in error for error in result["errors"])


def test_prepared_dataset_validation_rejects_out_of_order_and_nonfinite_features(tmp_path):
    dataset = tmp_path / "bad.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,return_1\n"
        "EURUSD,M15,2026-01-01T00:15:00Z,1,1.1,0.9,1.0,1,2026-01-01T00:30:00Z,0.1\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.0,0,2026-01-01T00:15:00Z,NaN\n",
        encoding="utf-8",
    )

    result = validate_prepared_dataset(dataset)

    assert result["status"] == "FAIL"
    assert any("timestamps_not_strictly_chronological" in error for error in result["errors"])
    assert any("invalid_or_missing_features" in error for error in result["errors"])


def test_experiment_summary_reports_fold_metrics_without_ranking_and_flags_overlap(tmp_path):
    manifest = _build_complete_run(tmp_path)
    matrix_path = json.loads(manifest.read_text(encoding="utf-8"))["experiment_matrix_report"]

    result = summarize_experiment_matrix(matrix_path)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["overlapping_test_window_count"] == 1
    summary = result["experiment_summaries"][0]
    assert summary["ranked_or_selected"] is False
    assert summary["metrics"]["brier_score"]["mean_across_folds"] == 0.21
    assert result["automatic_winner_selection"] is False


def test_completed_run_validation_writes_consolidated_non_actionable_report(tmp_path):
    manifest = _build_complete_run(tmp_path)
    output = tmp_path / "validation.json"

    result = validate_completed_run(manifest, output_path=output)

    assert output.is_file()
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["integrity_audit"]["status"] == "PASS"
    assert result["dataset_quality"]["status"] == "PASS"
    assert result["experiment_quality"]["status"] == "PASS_WITH_WARNINGS"
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["automatic_winner_selection"] is False


def test_ctrader_config_check_never_emits_secret_values(tmp_path, monkeypatch):
    manifest = _build_complete_run(tmp_path)
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path / "isolated-project")
    for name in ("CTRADER_HOST", "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    secret = "test-secret-value-do-not-print"
    monkeypatch.setenv("CTRADER_CLIENT_SECRET", secret)

    result = validate_completed_run(manifest, check_ctrader_config=True)

    serialized = json.dumps(result)
    assert result["ctrader_demo_config"]["status"] == "FAIL"
    assert result["status"] == "FAIL"
    assert result["ctrader_demo_config"]["credential_values_exposed"] is False
    assert secret not in serialized



def test_standalone_preflight_reports_only_presence_and_never_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path / "isolated-project")
    for name in ("CTRADER_HOST", "CTRADER_CLIENT_ID", "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    secret = "must-never-be-returned"
    monkeypatch.setenv("CTRADER_CLIENT_SECRET", secret)

    result = collect_research_preflight(check_ctrader_config=True)

    assert result["status"] == "FAIL"
    assert result["ctrader_demo_config"]["network_checked"] is False
    assert result["ctrader_demo_config"]["credential_values_exposed"] is False
    assert secret not in json.dumps(result)
    assert result["actionable"] is False



def test_research_preflight_reads_ctrader_values_from_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CTRADER_HOST=demo\n"
        "CTRADER_CLIENT_ID=dotenv-client-id\n"
        "CTRADER_CLIENT_SECRET=dotenv-client-secret\n"
        "CTRADER_ACCESS_TOKEN=dotenv-access-token\n"
        "CTRADER_ACCOUNT_ID=123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path)
    for name in ("CTRADER_HOST", "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET",
                 "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)

    result = collect_research_preflight(check_ctrader_config=True)

    assert result["ctrader_demo_config"]["status"] == "PASS"
    assert all(result["ctrader_demo_config"]["required_variables_present"].values())
    assert result["ctrader_demo_config"]["network_checked"] is False
    assert "dotenv-access-token" not in json.dumps(result)

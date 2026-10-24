from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import research_pipeline


def test_pipeline_requires_exactly_one_input_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="provider terms", source_version="test", output_dir=tmp_path,
            run_id="test-run", retrieved_at_utc="2026-10-21T00:00:00Z",
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )


def test_feature_names_excludes_contract_columns(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,ret_1,rsi_14\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1,1,1,1,2026-01-01T01:00:00Z,0.1,50\n",
        encoding="utf-8",
    )
    assert research_pipeline._feature_names_from_dataset(dataset) == ["ret_1", "rsi_14"]


def test_pipeline_orchestrates_offline_stages_and_marks_report_non_actionable(tmp_path, monkeypatch):
    raw = tmp_path / "ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )

    def fake_prepare(input_csv, output_csv, manifest_path, **kwargs):
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text(
            "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,ret_1\n"
            "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.1\n",
            encoding="utf-8",
        )
        Path(manifest_path).write_text('{"actionable":false}\n', encoding="utf-8")
        return {"feature_names": ["ret_1"], "actionable": False}

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", fake_prepare)
    monkeypatch.setattr(
        research_pipeline,
        "load_historical_dataset",
        lambda path, names: ([], [], "a" * 64),
    )
    monkeypatch.setattr(
        research_pipeline,
        "run_experiment_matrix",
        lambda *args, **kwargs: SimpleNamespace(fold_count=3, experiments=[{}, {}]),
    )

    def fake_write(report, path):
        Path(path).write_text('{"actionable":false}\n', encoding="utf-8")
        return str(path)

    monkeypatch.setattr(research_pipeline, "write_experiment_matrix_report", fake_write)

    result = research_pipeline.run_research_pipeline(
        symbol="EURUSD",
        timeframe="M15",
        horizon_bars=4,
        class_1_means="UP",
        terms_reference="actual provider terms",
        source_version="test-export-v1",
        output_dir=tmp_path / "run",
        run_id="unit-test",
        retrieved_at_utc="2026-10-21T00:00:00Z",
        ohlc_csv=raw,
        initial_train_bars=100,
        validation_bars=20,
        test_bars=20,
        step_bars=20,
        include_ablation=False,
    )
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["test_metrics_used_for_selection"] is False
    assert result["experiment_count"] == 2
    manifest = tmp_path / "run" / "unit-test_pipeline_manifest.json"
    assert manifest.exists()
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert Path(stored["dataset_path"]).name.startswith("unit-test_")
    assert Path(stored["dataset_manifest_path"]).name.startswith("unit-test_")
    assert stored["max_label_elapsed_hours"] is None
    assert stored["label_eligibility_mode"] == "all_labels_included"
    assert stored["actionable"] is False
    assert stored["raw_ohlc_sha256"]
    assert stored["raw_ohlc_audit_status"] == "PASS_WITH_WARNINGS"
    raw_audit_path = Path(stored["raw_ohlc_audit_path"])
    assert raw_audit_path.is_file()
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    assert raw_audit["errors"] == []
    assert "snapshot_manifest_missing_provenance_hash_unverified" in raw_audit["warnings"]
    assert stored["raw_ohlc_audit_sha256"]
    assert stored["dataset_sha256"]
    audit_path = tmp_path / "run" / "unit-test_provenance_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["source_report_sha256"] == stored["experiment_matrix_report_sha256"]
    assert stored["provenance_audit_sha256"]
    assert stored["provenance_audit_status"] == audit["status"] == "FAIL"
    assert stored["provenance_audit_errors"] == audit["errors"]
    assert stored["provenance_audit_warnings"] == audit["warnings"]
    assert stored["strict_provenance_audit"] is False
    temporal_audit_path = Path(stored["temporal_calibration_audit_path"])
    assert temporal_audit_path.is_file()
    temporal_audit = json.loads(temporal_audit_path.read_text(encoding="utf-8"))
    assert temporal_audit["scope"] == "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT"
    assert temporal_audit["experiment_matrix_sha256"] == stored["experiment_matrix_report_sha256"]
    assert stored["temporal_calibration_audit_sha256"] == research_pipeline._sha256(temporal_audit_path)
    assert stored["temporal_calibration_audit_status"] == temporal_audit["status"] == "FAIL"
    assert stored["temporal_calibration_audit_errors"] == temporal_audit["errors"]
    assert stored["temporal_calibration_audit_warnings"] == temporal_audit["warnings"]
    assert stored["strict_temporal_calibration_audit"] is False
    assert result["validation_status"] == "FAIL"  # deliberately incomplete mocked experiment report
    assert Path(result["validation_report_path"]).is_file()
    validation_report = json.loads(Path(result["validation_report_path"]).read_text(encoding="utf-8"))
    assert validation_report["actionable"] is False


def test_pipeline_requires_existing_ohlc_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="provider terms", source_version="test", output_dir=tmp_path,
            run_id="test-run", retrieved_at_utc="2026-10-21T00:00:00Z",
            ohlc_csv=tmp_path / "missing.csv",
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )



def test_pipeline_rejects_unsafe_run_id_before_writing_files(tmp_path):
    source = tmp_path / "ohlc.csv"
    source.write_text(
        "timestamp,open,high,low,close\n"
        "2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe filename token"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=1, class_1_means="UP",
            terms_reference="provider terms", source_version="test", output_dir=tmp_path / "run",
            run_id="../escape", retrieved_at_utc="2026-10-21T00:00:00Z",
            ohlc_csv=source, initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )
    assert not (tmp_path / "run").exists()



def test_pipeline_stops_before_dataset_preparation_when_raw_audit_fails(tmp_path, monkeypatch):
    raw = tmp_path / "bad_ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,0.9,0.8,0.95\n",
        encoding="utf-8",
    )

    def should_not_prepare(*args, **kwargs):
        pytest.fail("dataset preparation must not run when raw OHLC audit fails")

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", should_not_prepare)

    with pytest.raises(ValueError, match="raw OHLC audit failed before feature preparation"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="actual provider terms", source_version="test-export-v1",
            output_dir=tmp_path / "run", run_id="raw-audit-gate",
            retrieved_at_utc="2026-10-23T00:00:00Z", ohlc_csv=raw,
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )

    report_path = tmp_path / "run" / "raw-audit-gate_raw_ohlc_audit.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert any(error.startswith("invalid_ohlc_rows:") for error in report["errors"])



def test_pipeline_reuses_ctrader_manifest_provenance_for_local_snapshot(tmp_path, monkeypatch):
    raw = tmp_path / "EURUSD_M365_demo.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )
    retrieved_at = "2026-10-23T11:00:00Z"
    raw_manifest = {
        "schema_version": 1,
        "scope": "CTRADER_OPEN_API_HISTORICAL_OHLC_NORMALIZATION",
        "host": "demo",
        "actionable": False,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "rows": 1,
        "first_timestamp_utc": "2026-01-01T00:00:00Z",
        "last_timestamp_utc": "2026-01-01T00:00:00Z",
        "snapshot_sha256": research_pipeline._sha256(raw),
        "retrieved_at_utc": retrieved_at,
        "pagination": {"complete": True, "window_count": 1, "pages_requested": 1},
    }
    Path(str(raw) + ".manifest.json").write_text(
        json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8"
    )
    captured = {}

    def stop_after_provenance_check(input_csv, output_csv, manifest_path, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop_after_provenance_check")

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", stop_after_provenance_check)

    with pytest.raises(RuntimeError, match="stop_after_provenance_check"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="https://help.ctrader.com/open-api/terms-of-use/",
            source_version="ctrader-open-api==0.9.2",
            output_dir=tmp_path / "run", run_id="manifest-provenance-test",
            retrieved_at_utc=None, ohlc_csv=raw,
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
            include_ablation=False,
        )

    assert captured["data_source"] == "cTrader Open API (DEMO historical snapshot)"
    assert captured["retrieved_at_utc"] == retrieved_at

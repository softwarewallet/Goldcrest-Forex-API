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
    assert stored["actionable"] is False
    assert stored["raw_ohlc_sha256"]
    assert stored["dataset_sha256"]
    audit_path = tmp_path / "run" / "unit-test_provenance_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["source_report_sha256"] == stored["experiment_matrix_report_sha256"]
    assert stored["provenance_audit_sha256"]
    assert stored["provenance_audit_status"] == audit["status"] == "FAIL"
    assert stored["strict_provenance_audit"] is False


def test_pipeline_requires_existing_ohlc_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="provider terms", source_version="test", output_dir=tmp_path,
            run_id="test-run", retrieved_at_utc="2026-10-21T00:00:00Z",
            ohlc_csv=tmp_path / "missing.csv",
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )


def test_pipeline_manifest_hash_is_reproducible_and_covers_payload(tmp_path, monkeypatch):
    import hashlib

    # The pipeline manifest hash is computed from the canonical payload before its
    # own digest is inserted; verify that exact contract without invoking model training.
    raw = tmp_path / "ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )

    def fake_prepare(input_csv, output_csv, manifest_path, **kwargs):
        Path(output_csv).write_text(
            "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,ret_1\n"
            "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.1\n",
            encoding="utf-8",
        )
        Path(manifest_path).write_text('{"actionable":false}\n', encoding="utf-8")
        return {"actionable": False}

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", fake_prepare)
    monkeypatch.setattr(research_pipeline, "load_historical_dataset", lambda path, names: ([], [], "a" * 64))
    monkeypatch.setattr(research_pipeline, "run_experiment_matrix", lambda *args, **kwargs: SimpleNamespace(fold_count=0, experiments=[]))
    monkeypatch.setattr(research_pipeline, "write_experiment_matrix_report", lambda report, path: (Path(path).write_text('{"actionable":false}\n', encoding="utf-8") or str(path)))

    result = research_pipeline.run_research_pipeline(
        symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
        terms_reference="actual provider terms", source_version="test-export-v1",
        output_dir=tmp_path / "run", run_id="hash-check",
        retrieved_at_utc="2026-10-21T00:00:00Z", ohlc_csv=raw,
        initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        include_ablation=False,
    )
    manifest_path = tmp_path / "run" / "hash-check_pipeline_manifest.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = stored.pop("pipeline_manifest_sha256")
    canonical = json.dumps(stored, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest
    assert result["pipeline_manifest_sha256"] == digest


def test_strict_cli_returns_three_when_audit_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        research_pipeline,
        "run_research_pipeline",
        lambda **kwargs: {
            "run_id": "strict-fail-test",
            "provenance_audit_status": "FAIL",
            "actionable": False,
            "execution_connected": False,
        },
    )
    code = research_pipeline.main([
        "--ohlc-csv", "unused.csv",
        "--symbol", "EURUSD",
        "--timeframe", "M15",
        "--horizon-bars", "4",
        "--class-1-means", "UP",
        "--terms-reference", "provider terms",
        "--source-version", "test-export-v1",
        "--run-id", "strict-fail-test",
        "--initial-train-bars", "100",
        "--validation-bars", "20",
        "--test-bars", "20",
        "--step-bars", "20",
        "--strict-provenance-audit",
    ])
    captured = capsys.readouterr()
    assert code == 3
    assert "strict provenance audit failed" in captured.err
    assert '"execution_connected": false' in captured.out


def test_pipeline_stops_before_dataset_build_when_raw_quality_fails(tmp_path, monkeypatch):
    raw = tmp_path / "bad_ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,0.8,0.9,1.1\n",
        encoding="utf-8",
    )
    called = {"prepare": False}

    def should_not_prepare(*args, **kwargs):
        called["prepare"] = True
        raise AssertionError("dataset preparation must not run after a failed OHLC audit")

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", should_not_prepare)
    with pytest.raises(ValueError, match="historical OHLC quality audit failed"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="actual provider terms", source_version="test-export-v1",
            output_dir=tmp_path / "run", run_id="bad-data-run",
            retrieved_at_utc="2026-10-21T00:00:00Z", ohlc_csv=raw,
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )
    assert called["prepare"] is False
    receipt = tmp_path / "run" / "bad-data-run_data_quality_audit.json"
    assert receipt.exists()
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    assert stored["status"] == "FAIL"
    assert stored["actionable"] is False

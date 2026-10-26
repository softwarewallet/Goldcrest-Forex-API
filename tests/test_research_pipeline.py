from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import research_pipeline
from app.research_experiment_fingerprint import verify_experiment_fingerprint
from app.research_reproducibility_auditor import audit_experiment_reproducibility


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
        manifest = {
            "schema_version": 1,
            "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
            "actionable": False,
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "data_source": kwargs["data_source"],
            "source_version": kwargs["source_version"],
            "retrieved_at_utc": kwargs["retrieved_at_utc"],
            "terms_reference": kwargs["terms_reference"],
            "input_snapshot_sha256": research_pipeline._sha256(input_csv),
            "output_dataset_sha256": research_pipeline._sha256(output_csv),
            "output_rows": 1,
            "label_horizon_bars": kwargs["horizon_bars"],
            "class_1_means": kwargs["class_1_means"],
            "max_label_elapsed_hours": kwargs["max_label_elapsed_hours"],
            "feature_names": ["ret_1"],
            "feature_schema_sha256": hashlib.sha256(
                json.dumps(
                    {"feature_names": ["ret_1"]},
                    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "feature_implementation_file": "app/causal_ta_features.py",
            "feature_implementation_sha256": research_pipeline._sha256(
                Path(research_pipeline.__file__).with_name("causal_ta_features.py")
            ),
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        Path(manifest_path).write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(
        research_pipeline,
        "audit_point_in_time_invariance",
        lambda source, **kwargs: {
            "schema_version": 1,
            "audit_name": "research_point_in_time_feature_invariance_audit",
            "scope": "OFFLINE_OHLC_PREFIX_INVARIANCE",
            "status": "PASS",
            "source_path": str(source),
            "source_sha256": research_pipeline._sha256(source),
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "rows": 1,
            "feature_count": 1,
            "feature_names": ["ret_1"],
            "indicator_backend": "test",
            "cutoff_rows_checked": [0],
            "cutoff_count": 1,
            "feature_values_compared": 1,
            "errors": [],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "model_training_performed": False,
            "automatic_model_selection": False,
        },
    )
    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", fake_prepare)
    monkeypatch.setattr(
        research_pipeline,
        "audit_label_maturity",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "scope": "GOLDCREST_FORWARD_LABEL_MATURITY_AUDIT",
            "status": "PASS", "errors": [], "warnings": [],
            "raw_ohlc_path": str(args[0]),
            "raw_ohlc_sha256": research_pipeline._sha256(args[0]),
            "dataset_path": str(args[1]),
            "dataset_sha256": research_pipeline._sha256(args[1]),
            "checked_label_rows": 1, "eligible_label_rows": 0, "ineligible_label_rows": 0,
            "actionable": False, "execution_connected": False,
            "automatic_model_selection": False,
        },
    )
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
    assert stored["point_in_time_feature_invariance_audit_status"] == "PASS"
    point_in_time_path = Path(stored["point_in_time_feature_invariance_audit_path"])
    point_in_time_report = json.loads(point_in_time_path.read_text(encoding="utf-8"))
    assert point_in_time_report["source_sha256"] == stored["raw_ohlc_sha256"]
    assert stored["point_in_time_feature_invariance_audit_sha256"] == research_pipeline._sha256(point_in_time_path)
    raw_audit_path = Path(stored["raw_ohlc_audit_path"])
    assert raw_audit_path.is_file()
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    assert raw_audit["errors"] == []
    assert "snapshot_manifest_missing_provenance_hash_unverified" in raw_audit["warnings"]
    assert stored["raw_ohlc_audit_sha256"]
    assert stored["dataset_sha256"]
    assert stored["feature_source_audit_status"] == "PASS"
    assert stored["feature_source_audit_errors"] == []
    feature_source_audit_path = Path(stored["feature_source_audit_path"])
    feature_source_audit = json.loads(feature_source_audit_path.read_text(encoding="utf-8"))
    assert feature_source_audit["status"] == "PASS"
    assert stored["feature_source_audit_sha256"] == research_pipeline._sha256(feature_source_audit_path)
    assert stored["feature_schema_sha256"] == feature_source_audit["feature_schema_sha256_declared"]
    assert stored["feature_implementation_sha256"] == feature_source_audit["feature_implementation_sha256_declared"]
    assert stored["experiment_fingerprint_status"] == "PASS"
    fingerprint_path = Path(stored["experiment_fingerprint_path"])
    fingerprint_receipt = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert verify_experiment_fingerprint(fingerprint_receipt)["status"] == "PASS"
    assert stored["experiment_fingerprint_sha256"] == fingerprint_receipt["experiment_fingerprint_sha256"]
    fingerprinted_modules = set(fingerprint_receipt["implementation_file_sha256"])
    assert {
        "app/walk_forward.py",
        "app/model_walk_forward.py",
        "app/research_model_evaluation_gate.py",
        "app/research_provenance_auditor.py",
        "app/temporal_calibration_stability.py",
        "app/research_validation.py",
        "app/research_reproducibility_auditor.py",
        "app/research_pipeline_completion_audit.py",
        "app/research_data_audit.py",
        "app/research_label_maturity_audit.py",
    }.issubset(fingerprinted_modules)
    assert stored["experiment_fingerprint_file_sha256"] == research_pipeline._sha256(fingerprint_path)
    assert result["reproducibility_audit_status"] == "PASS"
    reproducibility_path = Path(result["reproducibility_audit_path"])
    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    assert reproducibility["status"] == "PASS"
    assert reproducibility["errors"] == []
    assert result["reproducibility_audit_sha256"] == research_pipeline._sha256(reproducibility_path)
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
    evaluation_gate_path = Path(stored["model_evaluation_gate_path"])
    assert evaluation_gate_path.is_file()
    evaluation_gate = json.loads(evaluation_gate_path.read_text(encoding="utf-8"))
    assert evaluation_gate["status"] == "FAIL"  # deliberately incomplete mocked matrix
    assert stored["model_evaluation_gate_sha256"] == research_pipeline._sha256(evaluation_gate_path)
    assert stored["model_evaluation_gate_status"] == evaluation_gate["status"]
    assert stored["model_evaluation_gate_errors"] == evaluation_gate["errors"]
    assert stored["strict_model_evaluation_gate"] is False
    assert result["validation_status"] == "FAIL"  # deliberately incomplete mocked experiment report
    assert Path(result["validation_report_path"]).is_file()
    validation_report = json.loads(Path(result["validation_report_path"]).read_text(encoding="utf-8"))
    assert validation_report["actionable"] is False
    assert validation_report["temporal_calibration_stability"]["status"] == "FAIL"

    completion_path = Path(result["pipeline_completion_audit_path"])
    assert completion_path.is_file()
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["schema_version"] == 2
    assert completion["pipeline_manifest_sha256"] == stored["pipeline_manifest_sha256"]
    assert completion["validation_report_sha256"] == research_pipeline._sha256(
        Path(result["validation_report_path"])
    )
    assert completion["reproducibility_audit_sha256"] == research_pipeline._sha256(
        reproducibility_path
    )
    assert completion["model_evaluation_gate_sha256"] == research_pipeline._sha256(
        evaluation_gate_path
    )
    assert completion["model_qualified"] is False
    assert completion["deployment_authorized"] is False
    assert result["pipeline_completion_audit_status"] == completion["status"]

    dataset_path = Path(stored["dataset_path"])
    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered_audit = audit_experiment_reproducibility(
        fingerprint_path, manifest
    )
    assert tampered_audit["status"] == "FAIL"
    assert "artifact_hash_mismatch:dataset_sha256" in tampered_audit["errors"]


def test_pipeline_refuses_existing_run_id_before_mutating_artifacts(tmp_path, monkeypatch):
    raw = tmp_path / "ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    existing = output_dir / "immutable_experiment_matrix.json"
    existing.write_text("preserve original artifact", encoding="utf-8")

    def should_not_prepare(*args, **kwargs):
        pytest.fail("pipeline must detect artifact collision before dataset preparation")

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", should_not_prepare)

    with pytest.raises(ValueError, match="refusing to overwrite artifacts from an existing run"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="actual provider terms", source_version="test-export-v1",
            output_dir=output_dir, run_id="immutable",
            retrieved_at_utc="2026-10-23T00:00:00Z", ohlc_csv=raw,
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )

    assert existing.read_text(encoding="utf-8") == "preserve original artifact"
    assert not (output_dir / "immutable_raw_ohlc_audit.json").exists()


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
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n"
        "EURUSD,M15,2026-01-01T00:15:00Z,1.05,1.15,0.95,1.1\n",
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
        "rows": 2,
        "first_timestamp_utc": "2026-01-01T00:00:00Z",
        "last_timestamp_utc": "2026-01-01T00:15:00Z",
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


def test_pipeline_stops_before_training_when_feature_source_audit_fails(tmp_path, monkeypatch):
    raw = tmp_path / "ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n"
        "EURUSD,M15,2026-01-01T00:15:00Z,1.05,1.15,0.95,1.1\n",
        encoding="utf-8",
    )

    def fake_prepare(input_csv, output_csv, manifest_path, **kwargs):
        Path(output_csv).write_text(
            "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,ret_1\n"
            "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,2026-01-01T01:00:00Z,0.1\n",
            encoding="utf-8",
        )
        Path(manifest_path).write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", fake_prepare)
    monkeypatch.setattr(
        research_pipeline,
        "audit_label_maturity",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "scope": "GOLDCREST_FORWARD_LABEL_MATURITY_AUDIT",
            "status": "PASS", "errors": [], "warnings": [],
            "raw_ohlc_path": str(args[0]),
            "raw_ohlc_sha256": research_pipeline._sha256(args[0]),
            "dataset_path": str(args[1]),
            "dataset_sha256": research_pipeline._sha256(args[1]),
            "checked_label_rows": 1, "eligible_label_rows": 0, "ineligible_label_rows": 0,
            "actionable": False, "execution_connected": False,
            "automatic_model_selection": False,
        },
    )
    monkeypatch.setattr(
        research_pipeline,
        "audit_feature_source_provenance",
        lambda *args, **kwargs: {
            "status": "FAIL",
            "errors": ["feature_implementation_sha256_mismatch"],
            "actionable": False,
            "execution_connected": False,
            "model_training_performed": False,
            "automatic_model_selection": False,
        },
    )
    monkeypatch.setattr(
        research_pipeline,
        "run_experiment_matrix",
        lambda *args, **kwargs: pytest.fail("training must not start after a failed feature-source audit"),
    )

    with pytest.raises(ValueError, match="feature source provenance audit failed before model training"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD",
            timeframe="M15",
            horizon_bars=4,
            class_1_means="UP",
            terms_reference="actual provider terms",
            source_version="test-export-v1",
            output_dir=tmp_path / "run",
            run_id="feature-source-gate",
            retrieved_at_utc="2026-10-21T00:00:00Z",
            ohlc_csv=raw,
            initial_train_bars=100,
            validation_bars=20,
            test_bars=20,
            step_bars=20,
            include_ablation=False,
        )

    assert not (tmp_path / "run" / "feature-source-gate_experiment_matrix.json").exists()



def test_pipeline_stops_before_dataset_preparation_when_point_in_time_audit_fails(tmp_path, monkeypatch):
    raw = tmp_path / "ohlc.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )

    def should_not_prepare(*args, **kwargs):
        pytest.fail("dataset preparation must not run when point-in-time invariance fails")

    monkeypatch.setattr(research_pipeline, "prepare_research_dataset", should_not_prepare)
    monkeypatch.setattr(
        research_pipeline,
        "audit_point_in_time_invariance",
        lambda *args, **kwargs: {
            "status": "FAIL",
            "source_sha256": research_pipeline._sha256(raw),
            "errors": [{"code": "future_data_changed_past_feature"}],
        },
    )

    with pytest.raises(ValueError, match="point-in-time feature invariance audit failed before feature preparation"):
        research_pipeline.run_research_pipeline(
            symbol="EURUSD", timeframe="M15", horizon_bars=4, class_1_means="UP",
            terms_reference="actual provider terms", source_version="test-export-v1",
            output_dir=tmp_path / "run", run_id="point-in-time-gate",
            retrieved_at_utc="2026-10-23T00:00:00Z", ohlc_csv=raw,
            initial_train_bars=100, validation_bars=20, test_bars=20, step_bars=20,
        )

    report_path = tmp_path / "run" / "point-in-time-gate_point_in_time_feature_invariance.json"
    assert report_path.is_file()
    assert not (tmp_path / "run" / "point-in-time-gate_EURUSD_M15_research_dataset.csv").exists()


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
        research_pipeline._read_json_object(path)

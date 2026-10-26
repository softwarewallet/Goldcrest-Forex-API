"""Offline auditor for Goldcrest experiment-matrix provenance artifacts.

This module checks recorded structure and cross-experiment consistency only. It does not
recompute metrics, certify leakage-free data, rank models, or connect to execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from app.research_strict_json import strict_json_loads


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _fold_windows(experiment: Mapping[str, Any]) -> list[tuple[Any, Any, Any]]:
    folds = experiment.get("fold_results")
    if not isinstance(folds, list):
        return []
    return [
        (fold.get("fold_index"), fold.get("test_start_utc"), fold.get("test_end_utc"))
        for fold in folds
        if isinstance(fold, dict)
    ]


def audit_experiment_matrix(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic findings for an offline experiment matrix JSON object."""
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != 1:
        errors.append("unsupported_or_missing_schema_version")
    if not isinstance(payload.get("run_id"), str) or not payload.get("run_id", "").strip():
        errors.append("missing_run_id")
    if not _is_sha256(payload.get("dataset_sha256")):
        errors.append("invalid_dataset_sha256")
    for field in ("symbol", "timeframe"):
        if not isinstance(payload.get(field), str) or not payload.get(field, "").strip():
            errors.append(f"missing_{field}")

    if payload.get("actionable") is not False:
        errors.append("actionable_must_be_false")
    if payload.get("automatic_winner_selection") is not False:
        errors.append("automatic_winner_selection_must_be_false")
    if payload.get("test_metrics_used_for_selection") is not False:
        errors.append("test_metrics_used_for_selection_must_be_false")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        errors.append("experiments_must_be_a_list")
        experiments = []

    declared_fold_count = payload.get("fold_count")
    if isinstance(declared_fold_count, bool) or not isinstance(declared_fold_count, int) or declared_fold_count < 0:
        errors.append("invalid_fold_count")

    expected_windows = payload.get("common_fold_test_windows")
    normalized_expected: list[tuple[Any, Any, Any]] = []
    if not isinstance(expected_windows, list):
        errors.append("common_fold_test_windows_must_be_a_list")
    else:
        if isinstance(declared_fold_count, int) and not isinstance(declared_fold_count, bool) and len(expected_windows) != declared_fold_count:
            errors.append("common_fold_window_count_mismatch")
        for window in expected_windows:
            if not isinstance(window, dict):
                errors.append("invalid_common_fold_window_record")
                continue
            normalized_expected.append((
                window.get("fold_index"), window.get("test_start_utc"), window.get("test_end_utc")
            ))
            if not isinstance(window.get("fold_index"), int) or isinstance(window.get("fold_index"), bool):
                errors.append("common_fold_window_invalid_fold_index")
            if not isinstance(window.get("test_start_utc"), str) or not window.get("test_start_utc").strip() or not isinstance(window.get("test_end_utc"), str) or not window.get("test_end_utc").strip():
                errors.append("common_fold_window_missing_boundary")

    ids: set[str] = set()
    model_types: set[str] = set()
    for index, experiment in enumerate(experiments):
        prefix = f"experiment_{index}"
        if not isinstance(experiment, dict):
            errors.append(f"{prefix}_must_be_an_object")
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            errors.append(f"{prefix}_missing_experiment_id")
        elif experiment_id in ids:
            errors.append(f"{prefix}_duplicate_experiment_id")
        else:
            ids.add(experiment_id)

        model_type = experiment.get("model_type")
        if not isinstance(model_type, str) or not model_type.strip():
            errors.append(f"{prefix}_missing_model_type")
        else:
            model_types.add(model_type)

        features = experiment.get("feature_names")
        if not isinstance(features, list) or not features or not all(
            isinstance(feature, str) and feature.strip() for feature in features
        ):
            errors.append(f"{prefix}_invalid_feature_names")
        elif len(features) != len(set(features)):
            errors.append(f"{prefix}_duplicate_feature_names")

        if not _is_sha256(experiment.get("feature_schema_sha256")):
            errors.append(f"{prefix}_invalid_feature_schema_sha256")
        if experiment.get("test_metrics_used_for_tuning") is not False:
            errors.append(f"{prefix}_test_metrics_used_for_tuning_must_be_false")
        if experiment.get("actionable") is not False:
            errors.append(f"{prefix}_actionable_must_be_false")

        windows = _fold_windows(experiment)
        folds = experiment.get("fold_results")
        if not isinstance(folds, list):
            errors.append(f"{prefix}_fold_results_must_be_a_list")
            continue
        if isinstance(declared_fold_count, int) and not isinstance(declared_fold_count, bool) and len(folds) != declared_fold_count:
            errors.append(f"{prefix}_fold_count_mismatch")
        if len(windows) != len(folds):
            errors.append(f"{prefix}_invalid_fold_record")
        elif expected_windows and windows != normalized_expected:
            errors.append(f"{prefix}_test_windows_do_not_match_common_plan")

        for fold_index, fold in enumerate(folds):
            if not isinstance(fold, dict):
                continue
            if not isinstance(fold.get("fold_index"), int) or isinstance(fold.get("fold_index"), bool):
                errors.append(f"{prefix}_fold_{fold_index}_invalid_fold_index")
            for boundary in ("test_start_utc", "test_end_utc"):
                if not isinstance(fold.get(boundary), str) or not fold.get(boundary).strip():
                    errors.append(f"{prefix}_fold_{fold_index}_missing_{boundary}")
            if not isinstance(fold.get("best_params"), dict):
                errors.append(f"{prefix}_fold_{fold_index}_missing_best_params")
            if isinstance(fold.get("trials_requested"), bool) or not isinstance(
                fold.get("trials_requested"), int
            ) or fold.get("trials_requested") < 1:
                errors.append(f"{prefix}_fold_{fold_index}_invalid_trials_requested")
            completed = fold.get("trials_completed")
            if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
                errors.append(f"{prefix}_fold_{fold_index}_invalid_trials_completed")
            elif isinstance(fold.get("trials_requested"), int) and completed > fold["trials_requested"]:
                errors.append(f"{prefix}_fold_{fold_index}_completed_trials_exceed_requested")

    declared_models = payload.get("model_types")
    if not isinstance(declared_models, list) or not all(isinstance(model, str) and model.strip() for model in declared_models):
        errors.append("invalid_declared_model_types")
    elif len(declared_models) != len(set(declared_models)):
        errors.append("duplicate_declared_model_types")
    elif set(declared_models) != model_types:
        errors.append("declared_model_types_do_not_match_experiments")
    if not experiments:
        warnings.append("no_experiment_records_to_audit")
    if not expected_windows:
        warnings.append("no_common_fold_windows_to_verify")

    return {
        "schema_version": 1,
        "run_id": payload.get("run_id"),
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "experiment_count": len(experiments),
        "declared_fold_count": payload.get("fold_count"),
        "audited_model_types": sorted(model_types),
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Structural and provenance consistency only. This audit does not recompute metrics, "
            "prove absence of leakage, validate market-data rights, or establish profitability."
        ),
    }


def audit_experiment_matrix_file(source: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    """Audit a JSON report and optionally write a deterministic JSON audit receipt."""
    source_path = Path(source)
    raw = source_path.read_bytes()
    payload = _load_json_object(raw, source=str(source_path))
    result = audit_experiment_matrix(payload)
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

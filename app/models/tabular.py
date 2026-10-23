"""Strict, offline wrappers for LightGBM and XGBoost binary classifiers.

These wrappers expose research probabilities only. They do not define market labels,
calibrate probabilities, choose trade directions, or enable actionable predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from app.resource_profile import get_ml_thread_count

_PROBABILITY_SEMANTICS = "uncalibrated_probability_of_class_1_research_only"


@dataclass(frozen=True)
class BinaryModelOutput:
    model_name: str
    model_version: str
    probability_class_1: float
    probability_semantics: str = _PROBABILITY_SEMANTICS
    actionable: bool = False


def feature_schema_fingerprint(feature_names: Sequence[str]) -> str:
    names = tuple(feature_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_matrix(
    feature_rows: Sequence[Mapping[str, float]], feature_names: Sequence[str]
) -> np.ndarray:
    names = tuple(feature_names)
    feature_schema_fingerprint(names)
    if not feature_rows:
        raise ValueError("feature_rows must not be empty")
    expected = set(names)
    matrix: list[list[float]] = []
    for row_index, row in enumerate(feature_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"feature row {row_index} must be a mapping")
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
        values: list[float] = []
        for name in names:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"feature {name!r} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"feature {name!r} must be finite")
            values.append(number)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def _validated_labels(labels: Sequence[int], row_count: int) -> np.ndarray:
    if len(labels) != row_count:
        raise ValueError("labels length must match feature_rows")
    values: list[int] = []
    for label in labels:
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("binary research labels must be explicitly supplied as 0 or 1")
        values.append(int(label))
    if len(set(values)) != 2:
        raise ValueError("training labels must contain both classes 0 and 1")
    return np.asarray(values, dtype=np.int32)


def _write_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def _read_metadata(model_path: Path, expected_model_type: str) -> dict[str, Any]:
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("model artifact and its metadata sidecar are both required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_type") != expected_model_type:
        raise ValueError("artifact model type does not match requested model")
    names = metadata.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError("artifact feature schema is missing")
    if metadata.get("feature_schema_sha256") != feature_schema_fingerprint(names):
        raise ValueError("artifact feature schema fingerprint is invalid")
    if not isinstance(metadata.get("model_version"), str) or not metadata["model_version"]:
        raise ValueError("artifact model version is missing")
    return metadata


class LightGBMResearchModel:
    """LightGBM binary classifier for offline research, not an approved strategy."""

    model_type = "lightgbm"

    def __init__(self, feature_names: Sequence[str], model_version: str = "candidate-v1") -> None:
        self.feature_names = tuple(feature_names)
        self.feature_schema_sha256 = feature_schema_fingerprint(self.feature_names)
        self.model_version = model_version
        if not model_version:
            raise ValueError("model_version must not be empty")
        self._classifier: lgb.LGBMClassifier | None = None
        self._booster: lgb.Booster | None = None

    def fit(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        labels: Sequence[int],
        *,
        params: Mapping[str, Any] | None = None,
    ) -> "LightGBMResearchModel":
        matrix = _validated_matrix(feature_rows, self.feature_names)
        target = _validated_labels(labels, len(matrix))
        options: dict[str, Any] = {"n_estimators": 100, "verbosity": -1, "random_state": 42, "n_jobs": get_ml_thread_count()}
        if params:
            options.update(dict(params))
        classifier = lgb.LGBMClassifier(**options)
        named_matrix = pd.DataFrame(matrix, columns=list(self.feature_names))
        classifier.fit(named_matrix, target, feature_name=list(self.feature_names))
        self._classifier, self._booster = classifier, None
        return self

    def predict(self, feature_row: Mapping[str, float]) -> BinaryModelOutput:
        matrix = _validated_matrix([feature_row], self.feature_names)
        if self._classifier is not None:
            classes = list(self._classifier.classes_)
            if 1 not in classes:
                raise ValueError("trained model has no class 1")
            named_row = pd.DataFrame(matrix, columns=list(self.feature_names))
            probability = float(self._classifier.predict_proba(named_row)[0][classes.index(1)])
        elif self._booster is not None:
            probability = float(np.asarray(self._booster.predict(matrix)).reshape(-1)[0])
        else:
            raise RuntimeError("LightGBM research model is not trained or loaded")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("LightGBM returned an invalid probability")
        return BinaryModelOutput(self.model_type, self.model_version, probability)

    def save(self, model_path: str | Path) -> None:
        if self._classifier is None:
            raise RuntimeError("only a fitted LightGBM classifier can be saved")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._classifier.booster_.save_model(str(path))
        _write_metadata(path, self._metadata())

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "feature_schema_sha256": self.feature_schema_sha256,
            "probability_semantics": _PROBABILITY_SEMANTICS,
            "actionable": False,
        }

    @classmethod
    def load(cls, model_path: str | Path) -> "LightGBMResearchModel":
        path = Path(model_path)
        metadata = _read_metadata(path, cls.model_type)
        instance = cls(metadata["feature_names"], metadata["model_version"])
        instance._booster = lgb.Booster(model_file=str(path))
        return instance


class XGBoostResearchModel:
    """XGBoost binary classifier for offline research, not an approved strategy."""

    model_type = "xgboost"

    def __init__(self, feature_names: Sequence[str], model_version: str = "candidate-v1") -> None:
        self.feature_names = tuple(feature_names)
        self.feature_schema_sha256 = feature_schema_fingerprint(self.feature_names)
        self.model_version = model_version
        if not model_version:
            raise ValueError("model_version must not be empty")
        self._classifier: xgb.XGBClassifier | None = None

    def fit(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        labels: Sequence[int],
        *,
        params: Mapping[str, Any] | None = None,
    ) -> "XGBoostResearchModel":
        matrix = _validated_matrix(feature_rows, self.feature_names)
        target = _validated_labels(labels, len(matrix))
        options: dict[str, Any] = {
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "n_jobs": get_ml_thread_count(), "random_state": 42,
        }
        if params:
            options.update(dict(params))
        classifier = xgb.XGBClassifier(**options)
        classifier.fit(matrix, target)
        self._classifier = classifier
        return self

    def predict(self, feature_row: Mapping[str, float]) -> BinaryModelOutput:
        matrix = _validated_matrix([feature_row], self.feature_names)
        if self._classifier is None:
            raise RuntimeError("XGBoost research model is not trained or loaded")
        classes = list(self._classifier.classes_)
        if 1 not in classes:
            raise ValueError("trained model has no class 1")
        probability = float(self._classifier.predict_proba(matrix)[0][classes.index(1)])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("XGBoost returned an invalid probability")
        return BinaryModelOutput(self.model_type, self.model_version, probability)

    def save(self, model_path: str | Path) -> None:
        if self._classifier is None:
            raise RuntimeError("only a fitted XGBoost classifier can be saved")
        path = Path(model_path)
        if path.suffix.lower() not in {".json", ".ubj"}:
            raise ValueError("XGBoost artifact path must end in .json or .ubj")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._classifier.save_model(str(path))
        _write_metadata(path, self._metadata())

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "feature_schema_sha256": self.feature_schema_sha256,
            "probability_semantics": _PROBABILITY_SEMANTICS,
            "actionable": False,
        }

    @classmethod
    def load(cls, model_path: str | Path) -> "XGBoostResearchModel":
        path = Path(model_path)
        metadata = _read_metadata(path, cls.model_type)
        instance = cls(metadata["feature_names"], metadata["model_version"])
        classifier = xgb.XGBClassifier()
        classifier.load_model(str(path))
        instance._classifier = classifier
        return instance

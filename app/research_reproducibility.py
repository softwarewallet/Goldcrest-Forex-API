"""Canonical reproducibility metadata for offline research experiments.

Metadata is evidence about the declared runtime/configuration, not proof that a run is
reproducible across hardware or that the source dataset/features are correct.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping


def _json_safe(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value, context=context)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{context} mapping keys must be strings")
        return {key: _json_safe(value[key], context=f"{context}.{key}") for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, context=f"{context}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item, context=f"{context}[]") for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    # Support NumPy scalar/array configuration values without making NumPy mandatory
    # for this metadata module.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item(), context=context)
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist(), context=context)
        except (ValueError, TypeError):
            pass
    raise ValueError(
        f"{context} contains unsupported value type {type(value).__module__}.{type(value).__qualname__}; "
        "expose JSON-compatible configuration metadata"
    )


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qualified_type(value: Any) -> str:
    cls = value if inspect.isclass(value) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _source_sha256(value: Any) -> str | None:
    target = value if (inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value)) else type(value)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def snapshot_factory_configuration(factory: Callable[[], Any]) -> dict[str, Any]:
    """Fingerprint factory identity, defaults, and serializable closure configuration."""
    if isinstance(factory, partial):
        descriptor = {
            "kind": "functools.partial",
            "callable": f"{factory.func.__module__}.{getattr(factory.func, '__qualname__', type(factory.func).__qualname__)}",
            "args": _json_safe(factory.args, context="factory.args"),
            "keywords": _json_safe(factory.keywords or {}, context="factory.keywords"),
            "source_sha256": _source_sha256(factory.func),
        }
    else:
        descriptor = {
            "kind": "class" if inspect.isclass(factory) else "callable",
            "callable": f"{getattr(factory, '__module__', type(factory).__module__)}.{getattr(factory, '__qualname__', type(factory).__qualname__)}",
            "defaults": _json_safe(getattr(factory, "__defaults__", None), context="factory.defaults"),
            "keyword_defaults": _json_safe(getattr(factory, "__kwdefaults__", None), context="factory.keyword_defaults"),
            "source_sha256": _source_sha256(factory),
        }
        closure = getattr(factory, "__closure__", None)
        if closure:
            freevars = getattr(getattr(factory, "__code__", None), "co_freevars", ())
            closure_values = {}
            for name, cell in zip(freevars, closure):
                try:
                    closure_values[name] = _json_safe(cell.cell_contents, context=f"factory.closure.{name}")
                except ValueError as exc:
                    raise ValueError(
                        f"model factory {descriptor['callable']} has non-serializable closure configuration: {exc}"
                    ) from exc
            descriptor["closure"] = closure_values
    descriptor["configuration_sha256"] = _fingerprint(descriptor)
    return descriptor


def snapshot_model_configuration(model: Any) -> dict[str, Any]:
    """Capture the model's declared public config and implementation source fingerprint."""
    public_attributes = {
        key: value for key, value in vars(model).items()
        if not key.startswith("_") and key not in {"trained"}
    }
    try:
        normalized_attributes = _json_safe(public_attributes, context="model.public_configuration")
    except ValueError as exc:
        getter = getattr(model, "get_research_config", None)
        if not callable(getter):
            raise ValueError(
                f"model {_qualified_type(model)} has non-serializable public configuration; "
                "implement get_research_config() returning JSON-compatible values"
            ) from exc
        normalized_attributes = _json_safe(getter(), context="model.get_research_config")
    getter = getattr(model, "get_research_config", None)
    if callable(getter):
        normalized_attributes["explicit_research_config"] = _json_safe(
            getter(), context="model.get_research_config"
        )
    get_params = getattr(model, "get_params", None)
    if callable(get_params):
        try:
            normalized_attributes["estimator_params"] = _json_safe(
                get_params(deep=True), context="model.get_params"
            )
        except TypeError:
            normalized_attributes["estimator_params"] = _json_safe(
                get_params(), context="model.get_params"
            )
    descriptor = {
        "model_class": _qualified_type(model),
        "model_type": str(getattr(model, "model_type", type(model).__name__)),
        "model_version": str(getattr(model, "model_version", "unknown")),
        "public_configuration": normalized_attributes,
        "source_sha256": _source_sha256(model),
    }
    descriptor["configuration_sha256"] = _fingerprint(descriptor)
    return descriptor


def runtime_environment() -> dict[str, Any]:
    """Record Python/platform and versions of the quantitative stack when installed."""
    packages = ("numpy", "scikit-learn", "lightgbm", "xgboost")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "python_executable_basename": Path(sys.executable).name,
    }


def reproducibility_manifest(
    factories: Mapping[str, Callable[[], Any]],
    initial_models: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a stable model/environment manifest and aggregate fingerprint."""
    factory_config = {
        name: snapshot_factory_configuration(factory)
        for name, factory in sorted(factories.items())
    }
    model_config = {
        name: snapshot_model_configuration(model)
        for name, model in sorted(initial_models.items())
    }
    payload = {
        "schema_version": 1,
        "environment": runtime_environment(),
        "factory_configuration": factory_config,
        "initial_model_configuration": model_config,
        "limitations": [
            "Runtime metadata and source fingerprints do not guarantee bitwise reproducibility across hardware.",
            "A source snapshot does not authenticate the dataset or prove feature causality.",
            "Factories must expose all meaningful configuration through JSON-compatible closure/defaults or model metadata.",
        ],
        "actionable": False,
    }
    payload["manifest_sha256"] = _fingerprint(payload)
    return payload

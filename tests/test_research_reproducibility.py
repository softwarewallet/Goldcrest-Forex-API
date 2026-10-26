from __future__ import annotations

import copy

import pytest

from app.research_reproducibility_auditor import _load_json_object
from app.research_reproducibility import (
    reproducibility_manifest,
    snapshot_factory_configuration,
    snapshot_model_configuration,
    verify_reproducibility_manifest,
)


class ConfiguredModel:
    model_type = "configured"
    model_version = "test-v1"

    def __init__(self, depth: int):
        self.depth = depth


def factory_with_depth(depth: int):
    return lambda: ConfiguredModel(depth)


def test_factory_fingerprint_changes_when_closure_configuration_changes():
    shallow = snapshot_factory_configuration(factory_with_depth(2))
    deep = snapshot_factory_configuration(factory_with_depth(7))
    assert shallow["configuration_sha256"] != deep["configuration_sha256"]
    assert shallow["closure"]["depth"] == 2
    assert deep["closure"]["depth"] == 7


def test_model_fingerprint_includes_public_configuration_and_source():
    first = snapshot_model_configuration(ConfiguredModel(2))
    second = snapshot_model_configuration(ConfiguredModel(7))
    assert first["configuration_sha256"] != second["configuration_sha256"]
    assert first["public_configuration"]["depth"] == 2
    assert second["public_configuration"]["depth"] == 7
    assert first["source_sha256"] == second["source_sha256"]


def test_reproducibility_manifest_verifies_and_detects_nested_tampering():
    factories = {"candidate": factory_with_depth(3)}
    models = {"candidate": factories["candidate"]()}
    manifest = reproducibility_manifest(factories, models)
    assert verify_reproducibility_manifest(manifest) is True
    tampered = copy.deepcopy(manifest)
    tampered["initial_model_configuration"]["candidate"]["public_configuration"]["depth"] = 99
    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        verify_reproducibility_manifest(tampered)


def test_reproducibility_manifest_rejects_nested_fingerprint_tampering():
    factories = {"candidate": factory_with_depth(3)}
    models = {"candidate": factories["candidate"]()}
    manifest = reproducibility_manifest(factories, models)
    tampered = copy.deepcopy(manifest)
    tampered["factory_configuration"]["candidate"]["closure"]["depth"] = 99
    # Recompute the outer digest to prove nested fingerprints are checked separately.
    import hashlib
    import json
    payload = dict(tampered)
    payload.pop("manifest_sha256")
    tampered["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="configuration fingerprint mismatch"):
        verify_reproducibility_manifest(tampered)


def test_non_serializable_factory_closure_fails_closed():
    opaque = object()
    factory = lambda: opaque
    with pytest.raises(ValueError, match="non-serializable closure configuration"):
        snapshot_factory_configuration(factory)


def test_environment_manifest_records_quantitative_stack_versions():
    manifest = reproducibility_manifest(
        {"candidate": factory_with_depth(3)},
        {"candidate": ConfiguredModel(3)},
    )
    environment = manifest["environment"]
    assert environment["python_version"]
    assert environment["python_implementation"]
    assert set(environment["packages"]) == {"numpy", "scikit-learn", "lightgbm", "xgboost"}


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_reproducibility_reader_rejects_ambiguous_json(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="reproducibility.json")

from __future__ import annotations

from copy import deepcopy

import pytest

from app.research_artifact_inventory_spec_seal import (
    canonical_inventory_spec_bytes,
    create_inventory_spec_seal,
    verify_inventory_spec_seal,
    _read_object,
)


def spec():
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification",
        "required_artifacts": [
            {"artifact_id": "metrics", "path": "reports/metrics.json"},
            {"artifact_id": "equity", "path": "reports/equity.csv"},
        ],
    }


def test_inventory_spec_seal_round_trip_passes():
    payload = spec()
    seal = create_inventory_spec_seal(payload)
    report = verify_inventory_spec_seal(payload, seal)
    assert report["status"] == "PASS"
    assert report["inventory_spec_sha256"] == seal["inventory_spec_sha256"]
    assert report["deployment_authorized"] is False


def test_canonicalization_is_independent_of_key_order():
    payload = spec()
    reordered = dict(reversed(list(payload.items())))
    assert canonical_inventory_spec_bytes(payload) == canonical_inventory_spec_bytes(reordered)
    assert create_inventory_spec_seal(payload)["inventory_spec_sha256"] == create_inventory_spec_seal(reordered)["inventory_spec_sha256"]


def test_inventory_spec_edit_after_sealing_is_detected():
    payload = spec()
    seal = create_inventory_spec_seal(payload)
    changed = deepcopy(payload)
    changed["required_artifacts"].append({"artifact_id": "extra", "path": "reports/extra.json"})
    report = verify_inventory_spec_seal(changed, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "inventory_spec_sha256" for item in report["errors"])


def test_seal_metadata_tampering_is_detected():
    payload = spec()
    seal = create_inventory_spec_seal(payload)
    seal["required_artifact_count"] = 99
    report = verify_inventory_spec_seal(payload, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "required_artifact_count" for item in report["errors"])


def test_missing_safety_flag_fails():
    payload = spec()
    seal = create_inventory_spec_seal(payload)
    del seal["execution_connected"]
    report = verify_inventory_spec_seal(payload, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "execution_connected" for item in report["errors"])


def test_invalid_inventory_spec_is_rejected():
    payload = spec()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version_unsupported"):
        create_inventory_spec_seal(payload)


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_json_reader_rejects_duplicate_keys_and_nonstandard_constants(tmp_path, raw):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("artifact_id", " metrics ", "artifact_id_must_be_canonical_unpadded"),
        ("path", " reports/metrics.json ", "path_must_be_canonical_unpadded"),
        ("path", "../metrics.json", "safe_normalized_posix"),
    ],
)
def test_inventory_spec_seal_rejects_noncanonical_artifact_identity(field, value, error):
    payload = spec()
    payload["required_artifacts"][0][field] = value
    with pytest.raises(ValueError, match=error):
        create_inventory_spec_seal(payload)

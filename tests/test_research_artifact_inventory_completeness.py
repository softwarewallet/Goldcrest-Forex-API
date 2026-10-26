from __future__ import annotations

import pytest

from app.research_artifact_inventory_completeness import audit_artifact_inventory_completeness


FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def manifest(entries=None):
    entries = entries if entries is not None else [
        {"artifact_id": "metrics", "path": "reports/metrics.json"},
        {"artifact_id": "equity", "path": "reports/equity.csv"},
    ]
    return {
        "audit_name": "research_artifact_provenance_manifest",
        "artifact_count": len(entries),
        "artifacts": entries,
        **FLAGS,
    }


def spec(entries=None):
    entries = entries if entries is not None else [
        {"artifact_id": "metrics", "path": "reports/metrics.json"},
        {"artifact_id": "equity", "path": "reports/equity.csv"},
    ]
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification",
        "required_artifacts": entries,
    }


def test_exact_inventory_passes():
    report = audit_artifact_inventory_completeness(manifest(), spec())
    assert report["status"] == "PASS"
    assert report["readiness"] == "INVENTORY_COMPLETE"
    assert report["manifest_artifact_count"] == 2
    assert report["inventory_spec_sha256"]
    assert report["deployment_authorized"] is False


def test_missing_required_artifact_fails():
    report = audit_artifact_inventory_completeness(
        manifest([{"artifact_id": "metrics", "path": "reports/metrics.json"}]), spec()
    )
    assert report["status"] == "FAIL"
    assert report["missing_or_mislocated_artifacts"] == [
        {"artifact_id": "equity", "path": "reports/equity.csv"}
    ]


def test_mislocated_artifact_fails():
    report = audit_artifact_inventory_completeness(
        manifest([{"artifact_id": "metrics", "path": "metrics.json"},
                  {"artifact_id": "equity", "path": "reports/equity.csv"}]), spec()
    )
    assert report["status"] == "FAIL"
    assert any(item["code"] == "REQUIRED_ARTIFACTS_MISSING_OR_MISLOCATED" for item in report["errors"])


def test_unexpected_artifact_fails_strict_inventory():
    entries = [
        {"artifact_id": "metrics", "path": "reports/metrics.json"},
        {"artifact_id": "equity", "path": "reports/equity.csv"},
        {"artifact_id": "unreviewed", "path": "reports/extra.json"},
    ]
    report = audit_artifact_inventory_completeness(manifest(entries), spec())
    assert report["status"] == "FAIL"
    assert report["unexpected_or_mislocated_artifacts"] == [
        {"artifact_id": "unreviewed", "path": "reports/extra.json"}
    ]


def test_duplicate_ids_and_paths_fail():
    entries = [
        {"artifact_id": "metrics", "path": "reports/metrics.json"},
        {"artifact_id": "metrics", "path": "reports/metrics.json"},
    ]
    report = audit_artifact_inventory_completeness(manifest(entries), spec(entries))
    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "FAIL"
    assert "DUPLICATE_ARTIFACT_ID" in codes
    assert "DUPLICATE_ARTIFACT_PATH" in codes


@pytest.mark.parametrize("path", ["/tmp/metrics.json", "../metrics.json", "reports/../metrics.json", "reports\\metrics.json"])
def test_unsafe_inventory_path_is_rejected(path):
    with pytest.raises(ValueError, match="safe_relative"):
        audit_artifact_inventory_completeness(
            manifest([{"artifact_id": "metrics", "path": path}]),
            spec([{"artifact_id": "metrics", "path": path}]),
        )


def test_manifest_count_mismatch_fails():
    payload = manifest()
    payload["artifact_count"] = 99
    report = audit_artifact_inventory_completeness(payload, spec())
    assert report["status"] == "FAIL"
    assert any(item["code"] == "MANIFEST_ARTIFACT_COUNT_MISMATCH" for item in report["errors"])


def test_invalid_specification_fails_closed():
    payload = spec()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version_unsupported"):
        audit_artifact_inventory_completeness(manifest(), payload)

@pytest.mark.parametrize(
    "raw",
    [
        b'{"audit_name":"research_artifact_provenance_manifest","audit_name":"shadowed"}',
        b'{"metric":NaN}',
        b'{"metric":Infinity}',
        b'{"metric":-Infinity}',
    ],
)
def test_inventory_reader_rejects_ambiguous_or_nonstandard_json(tmp_path, raw):
    from app.research_artifact_inventory_completeness import _read_object

    path = tmp_path / "input.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError):
        _read_object(path)


def test_inventory_reader_requires_object_root(tmp_path):
    from app.research_artifact_inventory_completeness import _read_object

    path = tmp_path / "input.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="json_root_must_be_object"):
        _read_object(path)

@pytest.mark.parametrize(
    ("manifest_entries", "spec_entries", "expected_error"),
    [
        ([], [{"artifact_id": "metrics", "path": "reports/metrics.json"}], "manifest_artifacts_must_not_be_empty"),
        ([{"artifact_id": "metrics", "path": "reports/metrics.json"}], [], "inventory_spec_artifacts_must_not_be_empty"),
    ],
)
def test_empty_artifact_inventory_cannot_pass_completeness_gate(
    manifest_entries, spec_entries, expected_error
):
    with pytest.raises(ValueError, match=expected_error):
        audit_artifact_inventory_completeness(
            manifest(manifest_entries),
            spec(spec_entries),
        )


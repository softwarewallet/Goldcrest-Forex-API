from __future__ import annotations

import json

import pytest

from app.research_evaluation_certification_evidence_bundle import (
    REPORT_SPECS,
    build_evidence_bundle,
)
from app.research_run_final_certification import (
    _write_pair_atomically,
    run_final_certification,
)


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


def make_inputs(tmp_path):
    paths = {}
    for role, (audit_name, _) in REPORT_SPECS.items():
        payload = {
            "audit_name": audit_name,
            "status": "PASS",
            "plan_id": "plan-abc",
            "plan_sha256": "a" * 64,
            "registry_id": "registry-xyz",
            "errors": [],
            **FLAGS,
        }
        if role == "plan":
            payload.pop("deployment_authorized")
            payload["expected_runs"] = 3
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths[role] = path
    bundle = build_evidence_bundle(paths)
    readiness = {
        "audit_name": "research_reproducibility_readiness",
        "status": "PASS",
        "plan_id": "plan-abc",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-xyz",
        "errors": [],
        **FLAGS,
    }
    return paths, bundle, readiness


def test_one_call_verifies_bundle_and_final_chain(tmp_path):
    _, bundle, readiness = make_inputs(tmp_path)
    verification, final_report = run_final_certification(
        readiness, bundle, evidence_root=tmp_path
    )
    assert verification["status"] == "PASS"
    assert verification["reports_verified"] == len(REPORT_SPECS)
    assert final_report["status"] == "PASS"
    assert final_report["readiness"] == "FINAL_CERTIFICATION_CHAIN_VERIFIED"
    assert final_report["deployment_authorized"] is False


def test_tampered_input_fails_final_chain(tmp_path):
    paths, bundle, readiness = make_inputs(tmp_path)
    paths["metric_semantics"].write_text('{"tampered":true}', encoding="utf-8")
    verification, final_report = run_final_certification(
        readiness, bundle, evidence_root=tmp_path
    )
    assert verification["status"] == "FAIL"
    assert final_report["status"] == "FAIL"
    assert any(e["code"] == "input_report_not_passing" for e in final_report["errors"])


def test_pair_writer_refuses_to_overwrite(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="output_already_exists"):
        _write_pair_atomically(first, {"status": "PASS"}, second, {"status": "PASS"})
    assert first.read_text(encoding="utf-8") == "preserve"
    assert not second.exists()


def test_pair_writer_requires_distinct_paths(tmp_path):
    output = tmp_path / "same.json"
    with pytest.raises(ValueError, match="output_paths_must_be_distinct"):
        _write_pair_atomically(output, {"status": "PASS"}, output, {"status": "PASS"})
    assert not output.exists()


def test_pair_writer_writes_both_reports(tmp_path):
    first = tmp_path / "nested" / "verification.json"
    second = tmp_path / "nested" / "final.json"
    _write_pair_atomically(first, {"status": "PASS"}, second, {"status": "PASS"})
    assert json.loads(first.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(second.read_text(encoding="utf-8"))["status"] == "PASS"


def test_pair_writer_rolls_back_first_output_if_second_publish_fails(tmp_path, monkeypatch):
    import app.research_run_final_certification as module

    first = tmp_path / "nested" / "verification.json"
    second = tmp_path / "nested" / "final.json"
    real_link = module.os.link
    calls = 0

    def fail_on_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated_second_publish_failure")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", fail_on_second_link)
    with pytest.raises(OSError, match="simulated_second_publish_failure"):
        _write_pair_atomically(
            first, {"status": "PASS"}, second, {"status": "PASS"}
        )

    assert not first.exists()
    assert not second.exists()
    assert list(first.parent.glob(".*.tmp")) == []

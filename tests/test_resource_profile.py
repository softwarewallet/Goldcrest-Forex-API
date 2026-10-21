from __future__ import annotations

import json

import pytest

from app.resource_profile import build_resource_profile
from app.runtime_diagnostics import collect_runtime_diagnostics


_GIB = 1024 ** 3


def test_8_gib_profile_limits_parallelism_on_small_laptop():
    profile = build_resource_profile(
        cpu_count=8, total_memory_bytes=8 * _GIB, available_memory_bytes=3 * _GIB
    )
    assert profile.profile_name == "constrained_8gib"
    assert profile.ml_threads == 2
    assert any("one training process" in note for note in profile.recommendations)


def test_16_gib_profile_uses_balanced_thread_count():
    profile = build_resource_profile(cpu_count=8, total_memory_bytes=16 * _GIB)
    assert profile.profile_name == "balanced_16gib"
    assert profile.ml_threads == 4


def test_thread_override_is_clamped_to_cpu_count():
    profile = build_resource_profile(
        cpu_count=4, total_memory_bytes=32 * _GIB, thread_override=12
    )
    assert profile.ml_threads == 4
    assert any("clamped" in note for note in profile.recommendations)


@pytest.mark.parametrize("value", [0, -1, "bad", True])
def test_invalid_thread_override_is_rejected(value):
    with pytest.raises(ValueError):
        build_resource_profile(cpu_count=4, total_memory_bytes=8 * _GIB, thread_override=value)


def test_diagnostics_are_research_only_and_do_not_claim_gateway_health(monkeypatch):
    monkeypatch.setenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("LLAMA_MODEL_ID", "qwen2.5-3b")
    report = collect_runtime_diagnostics()
    assert report["actionable"] is False
    assert report["scope"] == "OFFLINE_RESEARCH_ONLY"
    assert report["llm_gateway"]["base_url_configured"] is True
    assert report["llm_gateway"]["model_id_configured"] is True
    assert report["llm_gateway"]["network_checked"] is False
    assert "credentials" in " ".join(report["notes"]).lower()
    json.dumps(report)

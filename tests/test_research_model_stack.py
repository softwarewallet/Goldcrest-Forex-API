from __future__ import annotations

import pytest

from app.models.tabular import BinaryModelOutput
from app.research_model_stack import (
    build_tabular_model_factories,
    compare_research_outputs,
)


def test_factories_register_both_installed_tabular_candidates():
    factories = build_tabular_model_factories(["rsi", "return_5"], model_version="test-v1")
    assert set(factories) == {"xgboost", "lightgbm"}
    assert factories["xgboost"]().model_type == "xgboost"
    assert factories["lightgbm"]().model_type == "lightgbm"


def test_factory_rejects_ambiguous_feature_schema():
    with pytest.raises(ValueError, match="unique"):
        build_tabular_model_factories(["rsi", "rsi"])


def test_comparison_reports_agreement_without_fusing_probabilities():
    outputs = {
        "xgboost": BinaryModelOutput("xgboost", "xgb-v1", 0.82),
        "lightgbm": BinaryModelOutput("lightgbm", "lgbm-v1", 0.19),
    }
    result = compare_research_outputs(
        outputs,
        llm_decision="BUY",
        llm_model_id="llama-gateway/qwen-orchestrated",
    )
    by_name = {item.model_name: item for item in result.comparisons}
    assert by_name["xgboost"].direction == "UP"
    assert by_name["xgboost"].agreement == "AGREE"
    assert by_name["lightgbm"].direction == "DOWN"
    assert by_name["lightgbm"].agreement == "DISAGREE"
    assert result.scope == "RESEARCH_ONLY"
    assert result.fusion_applied is False
    assert result.actionable is False
    assert all(item.actionable is False for item in result.comparisons)


def test_deadband_and_llm_neutral_are_reported_explicitly():
    result = compare_research_outputs(
        {"xgboost": BinaryModelOutput("xgboost", "x1", 0.51)},
        llm_decision="NO_TRADE",
        llm_model_id="single-gateway",
    )
    item = result.comparisons[0]
    assert item.direction == "FLAT"
    assert item.llm_direction == "FLAT"
    assert item.agreement == "TABULAR_NEUTRAL"


@pytest.mark.parametrize("decision", ["LONG", "UNKNOWN", ""])
def test_invalid_llm_decision_fails_closed(decision):
    with pytest.raises(ValueError, match="LLM decision"):
        compare_research_outputs(
            {"xgboost": BinaryModelOutput("xgboost", "x1", 0.8)},
            llm_decision=decision,
            llm_model_id="single-gateway",
        )


def test_requires_gateway_provenance():
    with pytest.raises(ValueError, match="llm_model_id"):
        compare_research_outputs(
            {"xgboost": BinaryModelOutput("xgboost", "x1", 0.8)},
            llm_decision="BUY",
            llm_model_id=" ",
        )

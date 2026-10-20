"""Research-only registry and comparison for tabular ML plus one LLM gateway.

XGBoost and LightGBM are evaluated as independent tabular candidates. LLM output is
compared as a separate qualitative/directional observation. No probabilities are
averaged, no model is auto-selected, and no output is actionable.

Qwen orchestration, when desired, belongs behind the configured single LLM gateway.
This module deliberately does not open a second Qwen connection or assume that a
plain llama.cpp server can orchestrate another model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from app.model_walk_forward import BinaryResearchModel
from app.models.tabular import (
    BinaryModelOutput,
    LightGBMResearchModel,
    XGBoostResearchModel,
)

Direction = Literal["UP", "DOWN", "FLAT"]
LlmDecision = Literal["BUY", "SELL", "NO_TRADE"]
PositiveClassDirection = Literal["UP", "DOWN"]


@dataclass(frozen=True)
class ModelDirectionComparison:
    model_name: str
    model_version: str
    probability_class_1: float
    probability_semantics: str
    direction: Direction
    llm_direction: Direction
    agreement: Literal["AGREE", "DISAGREE", "TABULAR_NEUTRAL", "LLM_NEUTRAL"]
    actionable: bool = False


@dataclass(frozen=True)
class ResearchStackComparison:
    scope: Literal["RESEARCH_ONLY"]
    llm_model_id: str
    comparisons: tuple[ModelDirectionComparison, ...]
    fusion_applied: bool = False
    actionable: bool = False
    interpretation: str = (
        "Independent model comparison only; no blended probability, model promotion, "
        "trade recommendation, or execution authorization."
    )


def build_tabular_model_factories(
    feature_names: Sequence[str],
    *,
    model_version: str = "candidate-v1",
) -> dict[str, Callable[[], BinaryResearchModel]]:
    """Return fresh XGBoost and LightGBM candidates for the existing walk-forward runner."""
    names = tuple(feature_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    return {
        "xgboost": lambda: XGBoostResearchModel(names, model_version=model_version),
        "lightgbm": lambda: LightGBMResearchModel(names, model_version=model_version),
    }


def _tabular_direction(
    probability: float,
    *,
    positive_class_direction: PositiveClassDirection,
    lower_threshold: float,
    upper_threshold: float,
) -> Direction:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be within [0, 1]")
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")
    if probability >= upper_threshold:
        return positive_class_direction
    if probability <= lower_threshold:
        return "DOWN" if positive_class_direction == "UP" else "UP"
    return "FLAT"


def _llm_direction(decision: str) -> Direction:
    normalized = decision.strip().upper()
    if normalized == "BUY":
        return "UP"
    if normalized == "SELL":
        return "DOWN"
    if normalized == "NO_TRADE":
        return "FLAT"
    raise ValueError("LLM decision must be BUY, SELL, or NO_TRADE")


def compare_research_outputs(
    tabular_outputs: Mapping[str, BinaryModelOutput],
    *,
    llm_decision: str,
    llm_model_id: str,
    positive_class_direction: PositiveClassDirection = "UP",
    lower_threshold: float = 0.45,
    upper_threshold: float = 0.55,
) -> ResearchStackComparison:
    """Compare independently produced tabular outputs with one LLM-gateway result.

    Thresholds are explicit fixed research settings, not learned or tuned here. The
    class-1 label meaning must be supplied by the dataset/target contract. LLM output
    is expected from the configured single gateway; if that gateway orchestrates
    Qwen internally, its orchestration is represented by the gateway's model ID.
    """
    if not tabular_outputs:
        raise ValueError("at least one tabular model output is required")
    if not isinstance(llm_model_id, str) or not llm_model_id.strip():
        raise ValueError("llm_model_id must be provided for provenance")
    llm_direction = _llm_direction(llm_decision)
    rows: list[ModelDirectionComparison] = []
    for name, output in sorted(tabular_outputs.items()):
        if not name.strip():
            raise ValueError("tabular model names must not be empty")
        probability = output.probability_class_1
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("tabular probability must be numeric")
        direction = _tabular_direction(
            float(probability),
            positive_class_direction=positive_class_direction,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        if direction == "FLAT":
            agreement = "TABULAR_NEUTRAL"
        elif llm_direction == "FLAT":
            agreement = "LLM_NEUTRAL"
        elif direction == llm_direction:
            agreement = "AGREE"
        else:
            agreement = "DISAGREE"
        rows.append(ModelDirectionComparison(
            model_name=name,
            model_version=output.model_version,
            probability_class_1=float(probability),
            probability_semantics=output.probability_semantics,
            direction=direction,
            llm_direction=llm_direction,
            agreement=agreement,
            actionable=False,
        ))
    return ResearchStackComparison(
        scope="RESEARCH_ONLY",
        llm_model_id=llm_model_id,
        comparisons=tuple(rows),
        fusion_applied=False,
        actionable=False,
    )



async def compare_with_llm_gateway(
    tabular_outputs: Mapping[str, BinaryModelOutput],
    *,
    snapshot: Mapping[str, object],
    client: object,
    positive_class_direction: PositiveClassDirection = "UP",
    lower_threshold: float = 0.45,
    upper_threshold: float = 0.55,
) -> ResearchStackComparison:
    """Call the existing single LLM gateway and compare it with tabular candidates.

    Only a strict allowlist of point-in-time features is sent. The gateway client must
    expose async analyze(payload) -> (model_id, raw_text). A gateway that orchestrates
    Qwen internally can do so behind this same endpoint; this function never opens a
    second Qwen connection.
    """
    import math

    allowed_features = {
        "last_close", "ema_9", "ema_21", "rsi_14", "atr_14",
        "return_over_window", "quantitative_regime",
    }
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping")
    symbol = snapshot.get("symbol")
    timeframe = snapshot.get("timeframe")
    snapshot_at = snapshot.get("snapshot_at")
    raw_features = snapshot.get("features")
    if not isinstance(symbol, str) or not symbol or len(symbol) > 16:
        raise ValueError("snapshot symbol is invalid")
    if not isinstance(timeframe, str) or timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}:
        raise ValueError("snapshot timeframe is invalid")
    if not isinstance(snapshot_at, str) or not snapshot_at.strip():
        raise ValueError("snapshot_at must be an explicit timestamp string")
    if not isinstance(raw_features, Mapping):
        raise ValueError("snapshot features must be a mapping")
    safe_features: dict[str, float | str] = {}
    for name, value in raw_features.items():
        if name not in allowed_features:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                safe_features[name] = number
        elif name == "quantitative_regime" and isinstance(value, str):
            safe_features[name] = value[:40]
    if not safe_features:
        raise ValueError("snapshot has no allowlisted finite features")

    analyze = getattr(client, "analyze", None)
    if not callable(analyze):
        raise ValueError("client must expose async analyze(payload)")
    model_id, raw_output = await analyze({
        "symbol": symbol,
        "timeframe": timeframe,
        "snapshot_at": snapshot_at,
        "features": safe_features,
        "research_scope": "RESEARCH_ONLY",
        "actionable": False,
    })
    if not isinstance(model_id, str) or not model_id.strip() or not isinstance(raw_output, str):
        raise ValueError("LLM gateway returned invalid model provenance or output")
    from app.prediction_service import _parse_model_output

    decision, _regime, _reason_codes = _parse_model_output(raw_output)
    return compare_research_outputs(
        tabular_outputs,
        llm_decision=decision.value,
        llm_model_id=model_id,
        positive_class_direction=positive_class_direction,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

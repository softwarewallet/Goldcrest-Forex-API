import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import Settings
from app.features import calculate_features
from app.llama_client import LlamaClient
from app.models import AuditEvent, PredictionRecord
from app.schemas import Decision, MarketRegime, PredictionRequest, PredictionResponse

_ALLOWED_REASONS = {
    "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE", "TREND_CONFIRMED",
    "MOMENTUM_CONFIRMED", "TREND_UNCLEAR", "MODEL_OUTPUT_INVALID",
    "MODEL_DISAGREES_WITH_BASELINE", "BASELINE_ONLY_NO_VALIDATED_EDGE",
}


def _parse_model_output(raw: str) -> tuple[Decision, MarketRegime, list[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in model response")
    data = json.loads(text[start:end + 1])
    decision = Decision(data.get("decision", "NO_TRADE"))
    regime = MarketRegime(data.get("market_regime", "UNCERTAIN"))
    codes = data.get("reason_codes", [])
    if not isinstance(codes, list) or not all(isinstance(x, str) for x in codes):
        raise ValueError("Invalid reason codes")
    reasons = []
    for code in codes[:8]:
        cleaned = "".join(ch for ch in code.upper() if ch.isalnum() or ch == "_")[:48]
        if cleaned in _ALLOWED_REASONS:
            reasons.append(cleaned)
    return decision, regime, sorted(set(reasons)) or ["INSUFFICIENT_EVIDENCE"]


async def create_prediction(request: PredictionRequest, db: Session, settings: Settings, client: LlamaClient) -> PredictionResponse:
    now = datetime.now(timezone.utc)
    snapshot_time = request.market_data.snapshot_at
    age = (now - snapshot_time).total_seconds()
    if age > settings.max_snapshot_age_seconds:
        raise ValueError("Market snapshot is stale")
    if age < -10:
        raise ValueError("Market snapshot timestamp is in the future")
    if len(request.market_data.candles) > settings.max_candle_count:
        raise ValueError("Too many candles")

    features = calculate_features(request.market_data.candles)
    safe_payload = {
        "symbol": request.symbol, "timeframe": request.timeframe,
        "snapshot_at": snapshot_time.isoformat(), "bid": request.market_data.bid,
        "ask": request.market_data.ask, "spread": request.market_data.ask - request.market_data.bid,
        "requested_horizon_candles": request.requested_horizon_candles,
        "features": features, "candle_count": len(request.market_data.candles),
        "latest_candles": [c.model_dump(mode="json") for c in request.market_data.candles[-30:]],
    }

    model_id = None
    raw_output = None
    reasons = ["INSUFFICIENT_EVIDENCE"]
    regime = MarketRegime.UNCERTAIN
    try:
        model_id, raw_output = await client.analyze(safe_payload)
        model_decision, model_regime, model_reasons = _parse_model_output(raw_output)
        baseline = str(features["quantitative_regime"])
        if model_regime.value != baseline:
            reasons = ["MODEL_DISAGREES_WITH_BASELINE"]
        elif model_decision == Decision.NO_TRADE:
            reasons = model_reasons
        else:
            # Do not make LLM-only BUY/SELL output actionable before validating a quantitative rule.
            reasons = ["BASELINE_ONLY_NO_VALIDATED_EDGE"]
        regime = model_regime if model_regime.value == baseline else MarketRegime.UNCERTAIN
    except (ValueError, json.JSONDecodeError):
        reasons = ["MODEL_OUTPUT_INVALID"]
    except Exception as exc:
        # Never return internal network details or snapshot data in exception text.
        db.add(AuditEvent(event_type="inference", request_id=request.request_id, symbol=request.symbol,
                          outcome="failed", details={"error_type": type(exc).__name__}))

    prediction_id = str(uuid4())
    expires = min(now + timedelta(seconds=60), snapshot_time + timedelta(seconds=120))
    response = PredictionResponse(
        prediction_id=prediction_id, request_id=request.request_id, symbol=request.symbol,
        timeframe=request.timeframe, decision=Decision.NO_TRADE, direction=None, probability=None,
        market_regime=regime, reason_codes=reasons, model_id=model_id, generated_at=now, expires_at=expires,
    )
    db.add(PredictionRecord(
        id=prediction_id, request_id=request.request_id, symbol=request.symbol, timeframe=request.timeframe,
        decision=response.decision.value, direction=None, probability=None, market_regime=regime.value,
        reason_codes=reasons, model_id=model_id, generated_at=now, expires_at=expires,
        input_snapshot=safe_payload, raw_model_output=raw_output,
    ))
    db.add(AuditEvent(event_type="prediction", request_id=request.request_id, symbol=request.symbol,
                      outcome=response.decision.value,
                      details={"prediction_id": prediction_id, "reason_codes": reasons}))
    db.commit()
    return response

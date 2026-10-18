from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import Base, _engine, get_db
from app.llama_client import InferenceUnavailable, LlamaClient
from app.models import AuditEvent, OutcomeRecord, PredictionRecord
from app.prediction_service import create_prediction
from app.schemas import HealthResponse, OutcomeRequest, PredictionRequest, PredictionResponse
from app.security import require_api_key


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=_engine)
    yield


app = FastAPI(title="Goldcrest Forex API", version="0.1.0",
              description="AI market-analysis API; no broker execution routes.", lifespan=lifespan)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings), db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "unavailable"
    try:
        models = await LlamaClient(settings).list_models()
        inference_status = "ok" if models else "no_models"
    except InferenceUnavailable:
        inference_status = "unavailable"
    overall = "ok" if db_status == "ok" and inference_status == "ok" else "degraded"
    return HealthResponse(status=overall, application=settings.app_name,
                          inference_status=inference_status, database_status=db_status)


@app.get("/api/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(settings: Settings = Depends(get_settings)):
    try:
        models = await LlamaClient(settings).list_models()
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail="Inference server unavailable") from exc
    return {"inference_server": settings.llama_base_url,
            "configured_model_id": settings.llama_model_id or None,
            "models": [{"id": item["id"], "owned_by": item.get("owned_by")} for item in models]}


@app.post("/api/v1/predictions", response_model=PredictionResponse,
          dependencies=[Depends(require_api_key)])
async def predict(payload: PredictionRequest, settings: Settings = Depends(get_settings),
                  db: Session = Depends(get_db)):
    existing = db.scalar(select(PredictionRecord).where(PredictionRecord.request_id == payload.request_id))
    if existing:
        return PredictionResponse(
            prediction_id=existing.id, request_id=existing.request_id, symbol=existing.symbol,
            timeframe=existing.timeframe, decision=existing.decision, direction=existing.direction,
            probability=existing.probability, market_regime=existing.market_regime,
            reason_codes=existing.reason_codes, model_id=existing.model_id,
            generated_at=existing.generated_at, expires_at=existing.expires_at,
        )
    try:
        return await create_prediction(payload, db, settings, LlamaClient(settings))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/predictions/{prediction_id}", response_model=PredictionResponse,
         dependencies=[Depends(require_api_key)])
async def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    item = db.get(PredictionRecord, prediction_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionResponse(
        prediction_id=item.id, request_id=item.request_id, symbol=item.symbol, timeframe=item.timeframe,
        decision=item.decision, direction=item.direction, probability=item.probability,
        market_regime=item.market_regime, reason_codes=item.reason_codes, model_id=item.model_id,
        generated_at=item.generated_at, expires_at=item.expires_at,
    )


@app.post("/api/v1/outcomes", dependencies=[Depends(require_api_key)])
async def record_outcome(payload: OutcomeRequest, db: Session = Depends(get_db)):
    prediction = db.get(PredictionRecord, payload.prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if db.scalar(select(OutcomeRecord).where(OutcomeRecord.prediction_id == payload.prediction_id)):
        raise HTTPException(status_code=409, detail="Outcome already recorded")
    if payload.observed_at > datetime.now(timezone.utc):
        raise HTTPException(status_code=422, detail="observed_at cannot be in the future")
    db.add(OutcomeRecord(id=str(uuid4()), prediction_id=payload.prediction_id,
                         outcome_label=payload.outcome_label, observed_at=payload.observed_at,
                         details=payload.details))
    db.add(AuditEvent(event_type="outcome", request_id=prediction.request_id, symbol=prediction.symbol,
                      outcome=payload.outcome_label, details={"prediction_id": payload.prediction_id}))
    db.commit()
    return {"status": "recorded", "prediction_id": payload.prediction_id}


@app.get("/api/v1/performance", dependencies=[Depends(require_api_key)])
async def performance(db: Session = Depends(get_db)):
    return {
        "predictions_total": len(db.scalars(select(PredictionRecord)).all()),
        "outcomes_recorded": len(db.scalars(select(OutcomeRecord)).all()),
        "note": "Directional metrics are not published until outcome definitions and scoring are finalized.",
    }


@app.get("/api/v1/audit", dependencies=[Depends(require_api_key)])
async def audit(limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db)):
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).all()
    return [{"id": row.id, "event_type": row.event_type, "request_id": row.request_id,
             "symbol": row.symbol, "outcome": row.outcome, "duration_ms": row.duration_ms,
             "details": row.details, "created_at": row.created_at} for row in rows]

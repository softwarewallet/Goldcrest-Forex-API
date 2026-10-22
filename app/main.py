from contextlib import asynccontextmanager
import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import Base, _engine, get_db
from app.llama_client import InferenceUnavailable, LlamaClient
from app.models import AuditEvent, OutcomeRecord, PredictionRecord, ResearchExperimentRun
from app.prediction_service import _parse_model_output, create_prediction, request_fingerprint
from app.schemas import HealthResponse, OutcomeRequest, PredictionRequest, PredictionResponse
from app.security import require_api_key


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=_engine)
    yield


app = FastAPI(title="Goldcrest Forex API", version="0.1.0",
              description="AI market-analysis API; no broker execution routes.", lifespan=lifespan)

# Keep browser access explicit; configure trusted dashboard origins through GOLDCREST_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def _as_utc(value: datetime) -> datetime:
    """SQLite may return naive datetimes even for timezone-aware columns."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _prediction_response(item: PredictionRecord) -> PredictionResponse:
    return PredictionResponse(
        prediction_id=item.id, request_id=item.request_id, symbol=item.symbol, timeframe=item.timeframe,
        decision=item.decision, direction=item.direction, probability=item.probability,
        market_regime=item.market_regime, reason_codes=item.reason_codes, model_id=item.model_id,
        generated_at=_as_utc(item.generated_at), expires_at=_as_utc(item.expires_at),
    )


def _count_rows(db: Session, model) -> int:
    return int(db.scalar(select(func.count()).select_from(model)) or 0)


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
    api_key_configured = settings.api_key_configured
    overall = "ok" if db_status == "ok" and inference_status == "ok" and api_key_configured else "degraded"
    return HealthResponse(status=overall, application=settings.app_name,
                          inference_status=inference_status, database_status=db_status,
                          api_key_configured=api_key_configured)


@app.get("/api/v1/readiness/inference", dependencies=[Depends(require_api_key)])
async def inference_readiness(settings: Settings = Depends(get_settings)):
    """Perform a small real completion to verify inference, not merely model listing."""
    try:
        model_id, raw = await LlamaClient(settings).analyze({"readiness_probe": True, "instructions": "Return JSON only."})
        _parse_model_output(raw)
    except (InferenceUnavailable, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Inference completion or response validation failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Inference readiness check failed") from exc
    return {"status": "ok", "model_id": model_id, "completion_valid": True}


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
    fingerprint = request_fingerprint(payload)
    existing = db.scalar(select(PredictionRecord).where(PredictionRecord.request_id == payload.request_id))
    if existing:
        if (existing.input_snapshot or {}).get("request_fingerprint") != fingerprint:
            raise HTTPException(status_code=409, detail="request_id already used with a different payload")
        return _prediction_response(existing)
    try:
        return await create_prediction(payload, db, settings, LlamaClient(settings))
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IntegrityError as exc:
        # A concurrent request may have inserted the same id after our initial lookup.
        db.rollback()
        existing = db.scalar(select(PredictionRecord).where(PredictionRecord.request_id == payload.request_id))
        if existing:
            if (existing.input_snapshot or {}).get("request_fingerprint") == fingerprint:
                return _prediction_response(existing)
            raise HTTPException(status_code=409, detail="request_id already used with a different payload") from exc
        raise HTTPException(status_code=503, detail="Prediction persistence failed; retry with the same request_id") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Prediction persistence failed; retry with the same request_id") from exc


@app.get("/api/v1/predictions", response_model=list[PredictionResponse],
         dependencies=[Depends(require_api_key)])
async def list_predictions(limit: int = Query(default=30, ge=1, le=100),
                           db: Session = Depends(get_db)):
    """Return recent stored prediction records for the research dashboard."""
    rows = db.scalars(
        select(PredictionRecord).order_by(PredictionRecord.generated_at.desc(), PredictionRecord.id.desc()).limit(limit)
    ).all()
    return [_prediction_response(item) for item in rows]


@app.get("/api/v1/predictions/{prediction_id}", response_model=PredictionResponse,
         dependencies=[Depends(require_api_key)])
async def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    item = db.get(PredictionRecord, prediction_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _prediction_response(item)


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
    try:
        db.commit()
    except IntegrityError as exc:
        # A concurrent submission may have recorded this prediction's outcome first.
        db.rollback()
        raise HTTPException(status_code=409, detail="Outcome already recorded") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Outcome persistence failed; retry the request") from exc
    return {"status": "recorded", "prediction_id": payload.prediction_id}


@app.get("/api/v1/performance", dependencies=[Depends(require_api_key)])
async def performance(db: Session = Depends(get_db)):
    # Use database-side counts rather than materializing all records as the research store grows.
    return {
        "predictions_total": _count_rows(db, PredictionRecord),
        "outcomes_recorded": _count_rows(db, OutcomeRecord),
        "note": "Directional metrics are not published until outcome definitions and scoring are finalized.",
    }


@app.get("/api/v1/research/summary", dependencies=[Depends(require_api_key)])
async def research_summary(db: Session = Depends(get_db)):
    """Read-only aggregate research telemetry; intentionally does not claim prediction accuracy."""
    predictions_total = _count_rows(db, PredictionRecord)
    outcomes_recorded = _count_rows(db, OutcomeRecord)

    decision_rows = db.execute(
        select(PredictionRecord.decision, func.count())
        .group_by(PredictionRecord.decision)
        .order_by(PredictionRecord.decision)
    ).all()
    outcome_rows = db.execute(
        select(OutcomeRecord.outcome_label, func.count())
        .group_by(OutcomeRecord.outcome_label)
        .order_by(OutcomeRecord.outcome_label)
    ).all()
    model_rows = db.execute(
        select(PredictionRecord.model_id, func.count())
        .group_by(PredictionRecord.model_id)
        .order_by(PredictionRecord.model_id)
    ).all()

    return {
        "schema_version": "1.0",
        "predictions_total": predictions_total,
        "outcomes_recorded": outcomes_recorded,
        "pending_outcomes": max(predictions_total - outcomes_recorded, 0),
        "outcome_coverage": round(outcomes_recorded / predictions_total, 4) if predictions_total else None,
        "decisions": [{"decision": key, "count": int(count)} for key, count in decision_rows],
        "outcomes": [{"label": key, "count": int(count)} for key, count in outcome_rows],
        "models": [{"model_id": key or "unknown", "count": int(count)} for key, count in model_rows],
        "accuracy_metrics_published": False,
        "note": "Counts describe stored research records only. They are not directional accuracy, calibrated probabilities, or profitability metrics.",
    }


@app.get("/api/v1/research/statistics", dependencies=[Depends(require_api_key)])
async def research_statistics(db: Session = Depends(get_db)):
    """Descriptive barrier-outcome statistics; not model accuracy or trading profitability."""
    rows = db.execute(
        select(OutcomeRecord.outcome_label, func.count())
        .where(OutcomeRecord.outcome_label.in_(["TARGET_FIRST", "STOP_FIRST"]))
        .group_by(OutcomeRecord.outcome_label)
    ).all()
    counts = {label: int(count) for label, count in rows}
    from app.research_statistics import target_first_rate_summary

    summary = target_first_rate_summary(
        target_first=counts.get("TARGET_FIRST", 0),
        stop_first=counts.get("STOP_FIRST", 0),
    )
    return {
        "schema_version": "1.0",
        **summary,
        "excluded_outcomes": ["AMBIGUOUS", "HORIZON_EXPIRED", "INSUFFICIENT_DATA", "INVALIDATED", "UNKNOWN"],
        "interpretation": "Descriptive target-hit share among recorded TARGET_FIRST/STOP_FIRST outcomes only. This is not prediction accuracy, calibrated probability, expected return, or profitability.",
    }


@app.get("/api/v1/research/model-comparison", dependencies=[Depends(require_api_key)])
async def research_model_comparison(
    minimum_resolved_samples: int = Query(default=30, ge=1, le=1000000),
    credible_level: float = Query(default=0.95, gt=0, lt=1),
    db: Session = Depends(get_db),
):
    """Compare resolved barrier outcomes by recorded model ID; descriptive research only."""
    from scipy.stats import beta, fisher_exact

    rows = db.execute(
        select(PredictionRecord.model_id, PredictionRecord.symbol, PredictionRecord.timeframe,
               OutcomeRecord.outcome_label, func.count())
        .join(OutcomeRecord, OutcomeRecord.prediction_id == PredictionRecord.id)
        .where(OutcomeRecord.outcome_label.in_(["TARGET_FIRST", "STOP_FIRST"]))
        .group_by(PredictionRecord.model_id, PredictionRecord.symbol,
                  PredictionRecord.timeframe, OutcomeRecord.outcome_label)
    ).all()
    # Keep comparisons stratified by instrument and timeframe; do not pool unlike datasets.
    grouped = {}
    for model_id, symbol, timeframe, label, count in rows:
        group_key = (symbol, timeframe)
        model_key = model_id or "unknown"
        grouped.setdefault(group_key, {}).setdefault(model_key, {"TARGET_FIRST": 0, "STOP_FIRST": 0})[label] += int(count)

    strata = []
    for (symbol, timeframe), model_counts in sorted(grouped.items()):
        models = {}
        for model_id, counts in sorted(model_counts.items()):
            target, stop = counts["TARGET_FIRST"], counts["STOP_FIRST"]
            n = target + stop
            if n:
                alpha = (1 - credible_level) / 2
                interval = [float(beta.ppf(alpha, target + 0.5, stop + 0.5)),
                            float(beta.ppf(1 - alpha, target + 0.5, stop + 0.5))]
                share = target / n
            else:
                interval, share = None, None
            models[model_id] = {
                "target_first": target, "stop_first": stop, "resolved_barrier_outcomes": n,
                "target_first_share": share, "credible_interval": interval,
                "eligible_for_inference": n >= minimum_resolved_samples,
            }
        eligible = [key for key, value in models.items() if value["eligible_for_inference"]]
        comparisons = []
        for i, model_a in enumerate(eligible):
            for model_b in eligible[i + 1:]:
                a, b = models[model_a], models[model_b]
                odds, p_value = fisher_exact(
                    [[a["target_first"], a["stop_first"]], [b["target_first"], b["stop_first"]]],
                    alternative="two-sided",
                )
                comparisons.append({
                    "model_a": model_a, "model_b": model_b, "odds_ratio": float(odds),
                    "p_value": float(p_value), "p_value_holm_adjusted": None,
                })
        order = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["p_value"])
        adjusted_floor = 0.0
        for rank, index in enumerate(order):
            adjusted_floor = max(adjusted_floor, min(1.0, comparisons[index]["p_value"] * (len(order) - rank)))
            comparisons[index]["p_value_holm_adjusted"] = adjusted_floor
        strata.append({"symbol": symbol, "timeframe": timeframe, "models": models, "comparisons": comparisons})

    return {
        "schema_version": "1.0",
        "minimum_resolved_samples": minimum_resolved_samples,
        "credible_level": credible_level,
        "strata": strata,
        "accuracy_metrics_published": False,
        "interpretation": "Descriptive TARGET_FIRST versus STOP_FIRST barrier outcomes only. Comparisons are stratified by symbol and timeframe. This is not prediction accuracy, calibrated probability, expected return, or profitability.",
        "excluded_outcomes": ["AMBIGUOUS", "HORIZON_EXPIRED", "INSUFFICIENT_DATA", "INVALIDATED", "UNKNOWN"],
    }



def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _experiment_response(run: ResearchExperimentRun) -> dict:
    return {
        "run_id": run.run_id,
        "fingerprint_sha256": run.fingerprint_sha256,
        "dataset_sha256": run.dataset_sha256,
        "symbol": run.symbol,
        "timeframe": run.timeframe,
        "protocol_version": run.protocol_version,
        "manifest": run.manifest,
        "fold_results": run.fold_results,
        "created_at": _as_utc(run.created_at),
        "actionable": False,
        "automatic_winner_selection": False,
    }


@app.post("/api/v1/research/experiments", dependencies=[Depends(require_api_key)])
async def register_research_experiment(payload: dict = Body(...), db: Session = Depends(get_db)):
    """Persist an immutable manifest from an offline research run; never promotes a model."""
    required = {"run_id", "dataset_sha256", "symbol", "timeframe", "protocol_version", "manifest"}
    missing = required - set(payload)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {sorted(missing)}")
    run_id, dataset_hash = payload["run_id"], payload["dataset_sha256"]
    symbol, timeframe, protocol = payload["symbol"], payload["timeframe"], payload["protocol_version"]
    manifest = payload["manifest"]
    if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 128:
        raise HTTPException(status_code=422, detail="run_id must be a non-empty string of at most 128 characters")
    if not _is_sha256(dataset_hash):
        raise HTTPException(status_code=422, detail="dataset_sha256 must be a 64-character SHA-256 hex digest")
    if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 32:
        raise HTTPException(status_code=422, detail="symbol must be a non-empty string of at most 32 characters")
    if not isinstance(timeframe, str) or not timeframe.strip() or len(timeframe) > 8:
        raise HTTPException(status_code=422, detail="timeframe must be a non-empty string of at most 8 characters")
    if not isinstance(protocol, str) or not protocol.strip() or len(protocol) > 64:
        raise HTTPException(status_code=422, detail="protocol_version must be a non-empty string of at most 64 characters")
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=422, detail="manifest must be a JSON object")
    if manifest.get("actionable") is True or manifest.get("automatic_winner_selection") is True or manifest.get("test_metrics_used_for_selection") is True:
        raise HTTPException(status_code=422, detail="Research registry rejects actionable or test-selected manifests")
    if manifest.get("dataset_sha256") not in (None, dataset_hash):
        raise HTTPException(status_code=422, detail="manifest dataset hash does not match dataset_sha256")
    experiments = manifest.get("experiments", [])
    if not isinstance(experiments, list):
        raise HTTPException(status_code=422, detail="manifest.experiments must be a list")
    fold_results = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise HTTPException(status_code=422, detail="Each experiment must be a JSON object")
        if experiment.get("actionable") is True or experiment.get("test_metrics_used_for_tuning") is True:
            raise HTTPException(status_code=422, detail="Experiment entries must be offline and must not tune on test metrics")
        folds = experiment.get("fold_results", [])
        if not isinstance(folds, list):
            raise HTTPException(status_code=422, detail="experiment.fold_results must be a list")
        fold_results.append({
            "experiment_id": str(experiment.get("experiment_id", "")),
            "model_type": str(experiment.get("model_type", "")),
            "feature_variant": str(experiment.get("feature_variant", "")),
            "fold_results": folds,
        })
    canonical = json.dumps({
        "dataset_sha256": dataset_hash.lower(), "symbol": symbol.strip().upper(),
        "timeframe": timeframe.strip().upper(), "protocol_version": protocol.strip(),
        "manifest": manifest,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    existing = db.scalar(select(ResearchExperimentRun).where(ResearchExperimentRun.run_id == run_id))
    if existing:
        if existing.fingerprint_sha256 == fingerprint:
            return _experiment_response(existing)
        raise HTTPException(status_code=409, detail="run_id already exists with different experiment content")
    by_fingerprint = db.scalar(select(ResearchExperimentRun).where(ResearchExperimentRun.fingerprint_sha256 == fingerprint))
    if by_fingerprint:
        return _experiment_response(by_fingerprint)
    record = ResearchExperimentRun(
        run_id=run_id, fingerprint_sha256=fingerprint, dataset_sha256=dataset_hash.lower(),
        symbol=symbol.strip().upper(), timeframe=timeframe.strip().upper(),
        protocol_version=protocol.strip(), manifest=manifest, fold_results=fold_results,
    )
    db.add(record)
    db.add(AuditEvent(event_type="research_experiment_registered", symbol=symbol.strip().upper(),
                      outcome="RECORDED", details={"run_id": run_id, "fingerprint_sha256": fingerprint,
                      "dataset_sha256": dataset_hash.lower(), "experiment_count": len(experiments)}))
    try:
        db.commit()
        db.refresh(record)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Concurrent experiment registration conflict; query the registry") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Research experiment persistence failed") from exc
    return _experiment_response(record)


@app.get("/api/v1/research/experiments", dependencies=[Depends(require_api_key)])
async def list_research_experiments(
    limit: int = Query(default=50, ge=1, le=200),
    symbol: str | None = Query(default=None, min_length=1, max_length=32),
    timeframe: str | None = Query(default=None, min_length=1, max_length=8),
    db: Session = Depends(get_db),
):
    query = select(ResearchExperimentRun).order_by(ResearchExperimentRun.created_at.desc(), ResearchExperimentRun.run_id).limit(limit)
    if symbol:
        query = query.where(ResearchExperimentRun.symbol == symbol.strip().upper())
    if timeframe:
        query = query.where(ResearchExperimentRun.timeframe == timeframe.strip().upper())
    rows = db.scalars(query).all()
    return {"schema_version": "1.0", "count": len(rows), "runs": [_experiment_response(row) for row in rows],
            "actionable": False, "note": "Immutable offline research records only; no model promotion or execution."}


@app.get("/api/v1/research/experiments/{run_id}", dependencies=[Depends(require_api_key)])
async def get_research_experiment(run_id: str, db: Session = Depends(get_db)):
    run = db.scalar(select(ResearchExperimentRun).where(ResearchExperimentRun.run_id == run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Research experiment not found")
    return _experiment_response(run)


@app.get("/api/v1/audit", dependencies=[Depends(require_api_key)])
async def audit(limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db)):
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).all()
    return [{"id": row.id, "event_type": row.event_type, "request_id": row.request_id,
             "symbol": row.symbol, "outcome": row.outcome, "duration_ms": row.duration_ms,
             "details": row.details, "created_at": _as_utc(row.created_at)} for row in rows]

"""Authenticated, read-only research endpoints backed by persisted outcomes."""
from fastapi import APIRouter, Depends
from scipy.stats import beta, fisher_exact
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import OutcomeRecord, PredictionRecord
from app.db import get_db
from app.security import require_api_key

router = APIRouter(prefix="/api/v1/research", tags=["research"], dependencies=[Depends(require_api_key)])

@router.get("/model-comparison")
def model_comparison(minimum_resolved_samples: int = 30, credible_level: float = 0.95, db: Session = Depends(get_db)):
    """Compare per-model barrier outcome shares; never returns actionable signals."""
    if minimum_resolved_samples < 1 or minimum_resolved_samples > 1000000:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="minimum_resolved_samples must be between 1 and 1000000")
    if not 0 < credible_level < 1:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="credible_level must be strictly between 0 and 1")
    rows = db.execute(
        select(PredictionRecord.model_id, OutcomeRecord.outcome_label, func.count())
        .join(OutcomeRecord, OutcomeRecord.prediction_id == PredictionRecord.id)
        .where(OutcomeRecord.outcome_label.in_(["TARGET_FIRST", "STOP_FIRST"]))
        .group_by(PredictionRecord.model_id, OutcomeRecord.outcome_label)
    ).all()
    counts = {}
    for model_id, label, count in rows:
        key = model_id or "unknown"
        counts.setdefault(key, {"TARGET_FIRST": 0, "STOP_FIRST": 0})[label] = int(count)
    summaries = {}
    for model_id, values in sorted(counts.items()):
        target, stop = values["TARGET_FIRST"], values["STOP_FIRST"]
        n = target + stop
        if n:
            alpha = (1 - credible_level) / 2
            interval = [float(beta.ppf(alpha, target + 0.5, stop + 0.5)), float(beta.ppf(1 - alpha, target + 0.5, stop + 0.5))]
            share = target / n
        else:
            interval, share = None, None
        summaries[model_id] = {"target_first": target, "stop_first": stop, "resolved_barrier_outcomes": n, "target_first_share": share, "credible_interval": interval, "eligible_for_inference": n >= minimum_resolved_samples}
    eligible = list(summaries)
    eligible = [key for key in eligible if summaries[key]["eligible_for_inference"]]
    comparisons = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            a, b = summaries[left], summaries[right]
            odds, p = fisher_exact([[a["target_first"], a["stop_first"]], [b["target_first"], b["stop_first"]]], alternative="two-sided")
            comparisons.append({"model_a": left, "model_b": right, "odds_ratio": float(odds), "p_value": float(p)})
    ordered = sorted(range(len(comparisons)), key=lambda idx: comparisons[idx]["p_value"])
    running = 0.0
    for rank, idx in enumerate(ordered):
        running = max(running, min(1.0, comparisons[idx]["p_value"] * (len(ordered) - rank)))
        comparisons[idx]["p_value_holm_adjusted"] = running
    return {"schema_version": "1.0", "minimum_resolved_samples": minimum_resolved_samples, "credible_level": credible_level, "models": summaries, "comparisons": comparisons, "method": "two-sided Fisher exact test with Holm-Bonferroni correction", "accuracy_metrics_published": False, "interpretation": "Descriptive TARGET_FIRST versus STOP_FIRST outcome statistics only; not prediction accuracy, calibrated probability, expected return, or profitability. Compare only like-for-like instruments, timeframes, horizons, periods, and evaluation protocols."}

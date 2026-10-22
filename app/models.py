from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, Float, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    request_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    decision: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_regime: Mapped[str] = mapped_column(String(32), default="UNCERTAIN")
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    model_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    input_snapshot: Mapped[dict] = mapped_column(JSON)
    raw_model_output: Mapped[str | None] = mapped_column(Text, nullable=True)


class OutcomeRecord(Base):
    __tablename__ = "prediction_outcomes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    prediction_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    outcome_label: Mapped[str] = mapped_column(String(32))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    request_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    outcome: Mapped[str] = mapped_column(String(32))
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ResearchExperimentRun(Base):
    """Immutable, content-addressed record of an offline research experiment."""

    __tablename__ = "research_experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    fingerprint_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    dataset_sha256: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)
    protocol_version: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSON)
    fold_results: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

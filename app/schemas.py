import json
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Symbol = Annotated[str, Field(min_length=3, max_length=16, pattern=r"^[A-Z0-9/_-]+$")]
Timeframe = Annotated[str, Field(pattern=r"^(M1|M5|M15|M30|H1|H4|D1)$")]


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Candle timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def check_ohlc(self):
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open, close and high")
        return self


class MarketData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bid: float = Field(gt=0, allow_inf_nan=False)
    ask: float = Field(gt=0, allow_inf_nan=False)
    snapshot_at: datetime
    candles: list[Candle] = Field(min_length=21, max_length=500)

    @field_validator("snapshot_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_quotes_and_candles(self):
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        timestamps = [c.timestamp for c in self.candles]
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError("candles must be ordered ascending and have unique timestamps")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    symbol: Symbol
    timeframe: Timeframe
    market_data: MarketData
    requested_horizon_candles: int = Field(default=12, ge=1, le=500)

    @model_validator(mode="after")
    def ensure_symbol_format(self):
        if self.symbol != self.symbol.upper():
            raise ValueError("symbol must be uppercase")
        return self


class Decision(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"


class PredictionResponse(BaseModel):
    schema_version: str = "1.0"
    prediction_id: str
    request_id: str
    symbol: str
    timeframe: str
    decision: Decision
    direction: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    market_regime: MarketRegime = MarketRegime.UNCERTAIN
    reason_codes: list[str]
    model_id: str | None = None
    generated_at: datetime
    expires_at: datetime


class HealthResponse(BaseModel):
    status: str
    application: str
    inference_status: str
    database_status: str


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str
    outcome_label: str = Field(pattern=r"^(TARGET_FIRST|STOP_FIRST|HORIZON_EXPIRED|INVALIDATED|UNKNOWN)$")
    observed_at: datetime
    details: dict = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def bound_details_size(cls, value: dict) -> dict:
        try:
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("details must contain valid JSON values") from exc
        if len(encoded.encode("utf-8")) > 8192:
            raise ValueError("details must not exceed 8 KiB when serialized")
        return value

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value

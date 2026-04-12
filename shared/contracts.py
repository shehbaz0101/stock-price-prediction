"""Shared Pydantic contracts used across all services."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Schema hash — computed once from FEATURE_KEYS at import time
# ---------------------------------------------------------------------------
FEATURE_KEYS: list[str] = [
    "close",
    "returns_1d",
    "returns_5d",
    "returns_20d",
    "volume_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "atr_14",
    "obv",
    "nifty50_return_1d",
    "sp500_return_1d",
    "vix_last",
    "ticker_return_1d",
]

_SCHEMA_HASH: str = hashlib.md5(json.dumps(sorted(FEATURE_KEYS)).encode()).hexdigest()[:8]


class DataSource(str, Enum):
    YFINANCE = "yfinance"
    POLYGON = "polygon"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class OHLCVTick(BaseModel):
    ticker: str
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource = DataSource.YFINANCE

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper()

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def ohlcv_consistency(self) -> "OHLCVTick":
        if not (self.high >= self.close >= self.low):
            raise ValueError(
                f"OHLCV inconsistency: high={self.high} close={self.close} low={self.low}"
            )
        if not (self.high >= self.open >= self.low):
            raise ValueError(
                f"OHLCV inconsistency: high={self.high} open={self.open} low={self.low}"
            )
        if self.close <= 0:
            raise ValueError("close price must be positive")
        return self


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class FeatureVector(BaseModel):
    ticker: str
    timestamp_utc: datetime
    schema_hash: str = _SCHEMA_HASH
    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Prediction  (drift_score has a default so it's always optional to supply)
# ---------------------------------------------------------------------------
class Prediction(BaseModel):
    ticker: str
    timestamp_utc: datetime
    predicted_close: float
    confidence_pct: float
    prediction_lower: float
    prediction_upper: float
    model_version: str
    model_name: str
    feature_version: str
    schema_hash: str = _SCHEMA_HASH
    drift_score: float = 0.0          # FIX: default so callers don't have to supply it

    @model_validator(mode="after")
    def ci_valid(self) -> "Prediction":
        if self.prediction_lower > self.predicted_close:
            raise ValueError("prediction_lower must be <= predicted_close")
        if self.prediction_upper < self.predicted_close:
            raise ValueError("prediction_upper must be >= predicted_close")
        return self


# ---------------------------------------------------------------------------
# Ingestion error (dead-letter queue payload)
# ---------------------------------------------------------------------------
class IngestionError(BaseModel):
    ticker: str
    raw_payload: dict[str, Any]
    error_type: str
    error_msg: str
    source: DataSource = DataSource.YFINANCE
    ts: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# Quality check / report
# ---------------------------------------------------------------------------
class QualityCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class DataQualityReport(BaseModel):
    ticker: str
    checks: list[QualityCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# LLM insight
# ---------------------------------------------------------------------------
class InsightCitation(BaseModel):
    source: str
    snippet: str
    relevance: float = 0.0


class InsightResponse(BaseModel):
    ticker: str
    sentiment: str          # "bullish" | "bearish" | "neutral"
    directional_signal: str # "buy" | "sell" | "hold"
    summary: str
    rationale: str
    citations: list[InsightCitation] = Field(default_factory=list)
    confidence: float = 0.5

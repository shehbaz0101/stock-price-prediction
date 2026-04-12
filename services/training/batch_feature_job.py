"""Batch feature job. Casts row.to_dict() keys to str to fix dict[Hashable,Any] error."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from shared.contracts import DataSource, FeatureVector, _SCHEMA_HASH
from services.ingestion.fetcher import MarketDataFetcher
from services.ingestion.validator import TickValidator
from services.feature_engineering.stream_processor import (
    RealTimeFeatureComputer,
    TickerStateBuffer,
)

log = logging.getLogger(__name__)


class FeatureStoreClient:
    """Minimal protocol — replaced at runtime by the real FeatureStoreClient."""
    async def store_online(self, ticker: str, fv: Any, ttl: int = 3600) -> None:
        pass


async def run_batch_for_ticker(ticker: str, store: FeatureStoreClient) -> int:
    fetcher   = MarketDataFetcher(ticker, source=DataSource.YFINANCE,
                                  period="6mo", interval="1d")
    df: pd.DataFrame = await fetcher.fetch_latest()

    if df.empty:
        log.warning("No data fetched for batch job", extra={"ticker": ticker})
        return 0

    validator = TickValidator()
    computer  = RealTimeFeatureComputer()
    buf       = TickerStateBuffer(ticker, max_size=200)
    rows_done = 0

    for _, row in df.iterrows():
        try:
            # FIX: row.to_dict() → dict[Hashable, Any].
            # Cast all keys to str → dict[str, Any] as validator expects.
            raw_payload: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
            raw_payload.setdefault("source", DataSource.YFINANCE.value)
            tick = validator.validate(raw_payload, ticker)
        except Exception as exc:
            log.warning("Batch validation failed",
                        extra={"ticker": ticker, "exc": str(exc)})
            continue

        if not buf.push(tick):
            continue

        if not buf.ready:
            continue

        try:
            fv: FeatureVector = computer.compute(buf, tick)
        except Exception as exc:
            log.error("Batch feature computation failed",
                      extra={"exc": str(exc), "ticker": ticker})
            continue

        await store.store_online(ticker, fv)
        rows_done += 1

    log.info("Batch complete",
             extra={"ticker": ticker, "rows_written": rows_done})
    return rows_done

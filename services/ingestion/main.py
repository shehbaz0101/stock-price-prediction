"""Ingestion service. BackgroundTasks has NO default — FastAPI injects it."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.responses import JSONResponse

from shared.config import get_kafka_settings
from shared.contracts import DataSource, IngestionError
from shared.kafka.producer import KafkaProducerClient
from services.ingestion.fetcher import MarketDataFetcher
from services.ingestion.validator import TickValidator
from services.ingestion.data_quality import DataQualityMonitor

log = logging.getLogger(__name__)

_active_tickers: set[str] = set()
_kafka_producer: KafkaProducerClient | None = None
_validator = TickValidator()
_dq_monitor = DataQualityMonitor()

_stats: dict[str, Any] = {
    "ticks_published": 0,
    "validation_errors": 0,
    "dlq_events": 0,
    "last_tick_at": None,
}


def _get_producer() -> KafkaProducerClient:
    if _kafka_producer is None:
        raise RuntimeError("Kafka producer not initialised")
    return _kafka_producer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kafka_producer
    kafka_cfg = get_kafka_settings()
    _kafka_producer = KafkaProducerClient(kafka_cfg.BOOTSTRAP_SERVERS)
    await _kafka_producer.start()
    yield
    await _kafka_producer.stop()


app = FastAPI(title="Ingestion Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/tickers")
async def list_tickers() -> dict[str, list[str]]:
    return {"tickers": sorted(_active_tickers)}


@app.post("/v1/tickers/{ticker}", tags=["control"])
async def add_ticker(ticker: str) -> dict[str, str]:
    t = ticker.upper()
    _active_tickers.add(t)
    log.info("Ticker added", extra={"ticker": t})
    return {"ticker": t, "status": "added"}


@app.delete("/v1/tickers/{ticker}", tags=["control"])
async def remove_ticker(ticker: str) -> dict[str, str]:
    t = ticker.upper()
    _active_tickers.discard(t)
    log.info("Ticker removed", extra={"ticker": t})
    return {"ticker": t, "status": "removed"}


@app.post("/v1/ingest/now", tags=["control"])
async def ingest_now(
    # FIX: BackgroundTasks MUST NOT have a default — FastAPI auto-injects it
    background_tasks: BackgroundTasks,
    tickers: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    """Trigger an immediate ingestion cycle."""
    targets = [t.upper() for t in tickers] if tickers else list(_active_tickers)
    for t in targets:
        background_tasks.add_task(_ingest_ticker, t)
    return {"triggered": targets}


async def _ingest_ticker(ticker: str) -> None:
    kafka_cfg = get_kafka_settings()
    fetcher = MarketDataFetcher(ticker, source=DataSource.YFINANCE)

    try:
        df = await fetcher.fetch_latest()
        if df.empty:
            log.warning("No data fetched", extra={"ticker": ticker})
            return

        for _, row in df.iterrows():
            # FIX: row.to_dict() → dict[str, Any] (cast keys to str)
            raw_payload: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
            raw_payload["source"] = fetcher.source.value

            try:
                tick = _validator.validate(raw_payload, ticker)
            except Exception as exc:
                _stats["validation_errors"] += 1
                log.warning("Validation failed, routing to DLQ",
                            extra={"ticker": ticker, "exc": str(exc)})
                err = IngestionError(
                    ticker=ticker,
                    raw_payload=raw_payload,
                    error_type=type(exc).__name__,
                    error_msg=str(exc),
                    source=fetcher.source,
                )
                await _get_producer().publish_to_dlq(kafka_cfg.DLQ_TOPIC, err)
                _stats["dlq_events"] += 1
                continue

            topic = f"{kafka_cfg.RAW_TICKS_TOPIC}.{tick.ticker.replace('.', '_')}"
            await _get_producer().publish(topic, tick, key=tick.ticker)
            _stats["ticks_published"] += 1
            _stats["last_tick_at"] = tick.timestamp_utc.isoformat()
            log.info("Tick published",
                     extra={"ticker": ticker, "close": tick.close, "topic": topic})

    except Exception as exc:
        log.error("Ingestion failed", extra={"ticker": ticker, "exc": str(exc)})

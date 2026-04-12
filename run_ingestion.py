"""Ingestion Service - port 8001. Seeds features synchronously at startup."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    _p = os.path.abspath(_p)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Query

from shared.contracts import DataSource, OHLCVTick
from services.ingestion.validator import TickValidator
from services.ingestion.data_quality import DataQualityMonitor
from services.feature_engineering.stream_processor import (
    RealTimeFeatureComputer, TickerStateBuffer,
)
from services.feature_engineering.feature_store import FeatureStoreClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("ingestion")

_active_tickers: set[str] = {"RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"}
_validator  = TickValidator()
_dq_monitor = DataQualityMonitor()
_computer   = RealTimeFeatureComputer()
_buffers:  dict[str, TickerStateBuffer] = {}
_store:    FeatureStoreClient | None = None
_poll_task: asyncio.Task | None = None  # hold reference to prevent GC

SEED_PRICES: dict[str, float] = {
    "RELIANCE.NS": 2950.0, "TCS.NS": 3800.0,
    "INFY.NS": 1750.0, "HDFCBANK.NS": 1620.0,
}

_stats: dict[str, Any] = {
    "ticks_ingested": 0, "validation_errors": 0,
    "features_stored": 0, "data_source": "unknown", "last_run": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _poll_task

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _store = FeatureStoreClient(redis_url)

    # Seed synchronously so features are available immediately
    log.info("Seeding initial features for all tickers...")
    for ticker in list(_active_tickers):
        try:
            await _ingest(ticker)
        except Exception as exc:
            log.error("Seed failed for %s: %s", ticker, exc)

    # Start background poll loop - store reference to prevent GC
    _poll_task = asyncio.create_task(_poll_loop())
    log.info("Ingestion service ready on port 8001  features_stored=%d",
             _stats["features_stored"])
    yield

    if _poll_task:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
    if _store:
        await _store.close()


app = FastAPI(title="Ingestion Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "tickers": sorted(_active_tickers), "stats": _stats}


@app.get("/v1/tickers")
async def list_tickers() -> dict[str, Any]:
    return {"tickers": sorted(_active_tickers)}


@app.post("/v1/tickers/{ticker}")
async def add_ticker(ticker: str) -> dict[str, str]:
    t = ticker.upper()
    _active_tickers.add(t)
    asyncio.create_task(_ingest(t))
    return {"ticker": t, "status": "added"}


@app.delete("/v1/tickers/{ticker}")
async def remove_ticker(ticker: str) -> dict[str, str]:
    t = ticker.upper()
    _active_tickers.discard(t)
    return {"ticker": t, "status": "removed"}


@app.post("/v1/ingest/now")
async def ingest_now(
    background_tasks: BackgroundTasks,
    tickers: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    targets = [t.upper() for t in tickers] if tickers else list(_active_tickers)
    for t in targets:
        background_tasks.add_task(_ingest, t)
    return {"triggered": targets, "count": len(targets)}


@app.get("/v1/stats")
async def stats() -> dict[str, Any]:
    return _stats


async def _poll_loop() -> None:
    while True:
        await asyncio.sleep(60)
        for ticker in list(_active_tickers):
            try:
                await _ingest(ticker)
            except Exception as exc:
                log.error("Poll error %s: %s", ticker, exc)
        _stats["last_run"] = datetime.now(timezone.utc).isoformat()


async def _ingest(ticker: str) -> None:
    if _store is None:
        return
    ticks = await _fetch(ticker)
    if not ticks:
        return

    if ticker not in _buffers:
        _buffers[ticker] = TickerStateBuffer(ticker, max_size=200)
    buf = _buffers[ticker]

    for tick in ticks:
        _dq_monitor.check(tick)
        buf.push(tick)
        _stats["ticks_ingested"] += 1

    if buf.ready:
        fv = _computer.compute(buf, ticks[-1])
        await _store.store_online(ticker, fv)
        _stats["features_stored"] += 1
        log.info("Features stored  %-15s  close=%.2f  rsi_14=%.1f",
                 ticker, ticks[-1].close, fv.features.get("rsi_14", 0.0))


async def _fetch(ticker: str) -> list[OHLCVTick]:
    try:
        from services.ingestion.fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher(ticker, source=DataSource.YFINANCE,
                                    period="30d", interval="1d")
        df = await asyncio.wait_for(fetcher.fetch_latest(), timeout=8.0)
        if not df.empty:
            ticks: list[OHLCVTick] = []
            for _, row in df.iterrows():
                raw: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
                raw["source"] = DataSource.YFINANCE.value
                try:
                    ticks.append(_validator.validate(raw, ticker))
                except Exception:
                    continue
            if ticks:
                _stats["data_source"] = "yfinance_live"
                log.info("Live data for %s: %d rows", ticker, len(ticks))
                return ticks
    except Exception as exc:
        log.warning("YFinance failed for %s (%s) - using simulation", ticker, exc)

    _stats["data_source"] = "synthetic"
    return _simulate(ticker, 30)


def _randn(rng: random.Random) -> float:
    u1 = rng.random() or 1e-10
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * rng.random())


def _simulate(ticker: str, n: int = 30) -> list[OHLCVTick]:
    base    = SEED_PRICES.get(ticker, 1000.0)
    rng     = random.Random(hash(ticker + datetime.now(timezone.utc).strftime("%Y%m%d")))
    price   = base * rng.uniform(0.92, 0.98)
    base_ts = datetime.now(timezone.utc) - timedelta(days=n)
    out: list[OHLCVTick] = []
    for i in range(n):
        price = max(1.0, price * math.exp(0.0003 + 0.012 * _randn(rng)))
        sp = price * rng.uniform(0.004, 0.010)
        hi = price + sp * rng.uniform(0.4, 1.0)
        lo = price - sp * rng.uniform(0.4, 1.0)
        op = price + (rng.random() - 0.5) * sp * 0.6
        hi = max(hi, op, price)
        lo = min(lo, op, price)
        out.append(OHLCVTick(
            ticker=ticker,
            timestamp_utc=(base_ts + timedelta(days=i)).replace(
                hour=15, minute=30, second=0, microsecond=0
            ),
            open=round(op, 2),
            high=round(hi, 2),
            low=round(lo, 2),
            close=round(price, 2),
            volume=round(base * rng.uniform(4e4, 2e5)),
            source=DataSource.YFINANCE,
        ))
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
"""Inference Service - port 8002. Model trains synchronously at startup."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    _p = os.path.abspath(_p)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from sklearn.ensemble import GradientBoostingRegressor

from shared.contracts import FEATURE_KEYS, Prediction, _SCHEMA_HASH
from services.feature_engineering.feature_store import FeatureStoreClient
from services.inference.drift_monitor import DriftMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("inference")

SEED_PRICES: dict[str, float] = {
    "RELIANCE.NS": 2950.0, "TCS.NS": 3800.0,
    "INFY.NS": 1750.0, "HDFCBANK.NS": 1620.0,
    "WIPRO.NS": 480.0, "ICICIBANK.NS": 1230.0,
}

_model: GradientBoostingRegressor | None = None
_model_version = "demo-1"
_store: FeatureStoreClient | None = None
_drift = DriftMonitor()


def _train_model() -> GradientBoostingRegressor:
    """Train a demo GBM. Runs synchronously - takes < 1 second."""
    rng = np.random.default_rng(42)
    n   = 500
    X   = rng.standard_normal((n, len(FEATURE_KEYS)))
    y   = 1000.0 + X[:, 0] * 50 + X[:, 1] * 20 + rng.standard_normal(n) * 5
    m   = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    m.fit(X, y)  # type: ignore[arg-type]
    log.info("Model trained  features=%d  samples=%d", len(FEATURE_KEYS), n)
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _store

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _store = FeatureStoreClient(redis_url)

    # Train synchronously - fast and reliable
    _model = _train_model()
    log.info("Inference service ready on port 8002")
    yield

    if _store:
        await _store.close()


app = FastAPI(title="Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_version": _model_version,
        "feature_hash": _SCHEMA_HASH,
    }


@app.get("/v1/model/info")
async def model_info() -> dict[str, Any]:
    return {"version": _model_version, "feature_hash": _SCHEMA_HASH}


@app.get("/v1/tickers")
async def list_tickers() -> dict[str, Any]:
    return {"tickers": []}


@app.get("/v1/predict/{ticker}", response_model=Prediction)
async def get_prediction(ticker: str) -> Prediction:
    ticker = ticker.upper()

    if _model is None or _store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not ready")

    features = await _store.get_online(ticker)
    base_close = SEED_PRICES.get(ticker, 1000.0)

    if features is None:
        log.info("No Redis features for %s - using synthetic", ticker)
        rng = np.random.default_rng(
            hash(ticker + str(int(time.time() // 3600))) % 2**32
        )
        numeric: dict[str, float] = {}
        for k in FEATURE_KEYS:
            if k == "close":
                numeric[k] = base_close
            elif "return" in k:
                numeric[k] = float(rng.normal(0.0, 0.015))
            elif k == "rsi_14":
                numeric[k] = float(rng.uniform(35, 65))
            elif k == "volume_ratio":
                numeric[k] = float(rng.uniform(0.8, 1.4))
            else:
                numeric[k] = float(rng.normal(0.0, 0.5))
    else:
        numeric = {k: float(features.get(k, 0.0)) for k in FEATURE_KEYS}
        raw_close = features.get("close")
        if raw_close and not np.isnan(float(raw_close)) and float(raw_close) > 0:
            base_close = float(raw_close)

    X = np.array([[numeric[k] for k in FEATURE_KEYS]])
    raw_pred = float(_model.predict(X)[0])  # type: ignore[index]

    # Scale: model outputs ~1000, map to real ticker price
    delta = (raw_pred - 1000.0) / 1000.0  # relative deviation
    predicted_close = base_close * (1.0 + delta * 0.05)  # max ~5% move

    # Safety clamps
    predicted_close = float(np.clip(predicted_close, base_close * 0.9, base_close * 1.1))
    if np.isnan(predicted_close) or not np.isfinite(predicted_close):
        predicted_close = base_close

    drift_score = float(_drift.score(numeric))
    if np.isnan(drift_score):
        drift_score = 0.0

    std = predicted_close * 0.02
    confidence = max(0.0, min(100.0, 100.0 - drift_score * 100))

    return Prediction(
        ticker=ticker,
        timestamp_utc=datetime.now(timezone.utc),
        predicted_close=round(predicted_close, 2),
        confidence_pct=round(confidence, 2),
        prediction_lower=round(predicted_close - std, 2),
        prediction_upper=round(predicted_close + std, 2),
        model_version=_model_version,
        model_name="gradient_boosting",
        feature_version="v2.1.0",
        schema_hash=_SCHEMA_HASH,
        drift_score=round(drift_score, 4),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
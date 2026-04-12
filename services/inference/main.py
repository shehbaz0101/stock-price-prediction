"""Inference service. _model_registry guarded for None before every access."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from shared.contracts import FeatureVector, Prediction, _SCHEMA_HASH
from services.inference.model_registry import MLflowModelRegistry
from services.inference.drift_monitor import DriftMonitor
from services.feature_engineering.feature_store import FeatureStoreClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — all start as None; initialised in lifespan
# ---------------------------------------------------------------------------
_model_registry: MLflowModelRegistry | None = None
_feature_store:  FeatureStoreClient  | None = None
_drift_monitor:  DriftMonitor        | None = None

MODEL_RELOAD_INTERVAL = int(os.getenv("MODEL_RELOAD_INTERVAL", "300"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_registry, _feature_store, _drift_monitor

    mlflow_uri  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name  = os.getenv("MLFLOW_MODEL_NAME",   "stock-predictor")
    redis_url   = os.getenv("REDIS_URL",            "redis://localhost:6379/0")

    _drift_monitor  = DriftMonitor()
    _feature_store  = FeatureStoreClient(redis_url)
    _model_registry = MLflowModelRegistry(mlflow_uri, model_name)
    _model_registry.sync_load()

    reload_task = asyncio.create_task(_model_reload_loop())

    yield

    reload_task.cancel()
    try:
        await reload_task
    except asyncio.CancelledError:
        pass
    if _feature_store:
        await _feature_store.close()


app = FastAPI(title="Inference Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------
async def _model_reload_loop() -> None:
    while True:
        await asyncio.sleep(MODEL_RELOAD_INTERVAL)
        try:
            # FIX: guard None before calling any method
            if _model_registry is None:
                continue
            reloaded = await _model_registry.reload_if_new()
            if reloaded:
                # FIX: only access .current_version after None check
                log.info("Model reloaded",
                         extra={"version": _model_registry.current_version})
        except Exception as exc:
            log.error("Model reload failed", extra={"exc": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok" if _model_registry is not None else "degraded"}


@app.get("/v1/model/info")
async def model_info() -> dict[str, Any]:
    if _model_registry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return {
        "version":      _model_registry.current_version,
        "feature_hash": _model_registry.feature_schema_hash,
    }


@app.get("/v1/predict/{ticker}", response_model=Prediction)
async def get_prediction(ticker: str) -> Prediction:
    """Fast prediction endpoint — returns ML prediction only.
    Latency target: <100 ms (cache hit) / <500 ms (cache miss).
    """
    ticker = ticker.upper()

    if _model_registry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    if _feature_store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Feature store not ready")

    features = await _feature_store.get_online(ticker)
    if features is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"No features found for {ticker}")

    numeric = {k: float(v) for k, v in features.items()
               if isinstance(v, (int, float)) and k not in ("ticker",)}

    if _model_registry.feature_schema_hash != _SCHEMA_HASH:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Feature schema mismatch — retrain required")

    predicted_close = _model_registry.predict(numeric)
    drift_score     = _drift_monitor.score(numeric) if _drift_monitor else 0.0
    std             = predicted_close * 0.03   # placeholder CI ±3 %

    return Prediction(
        ticker=ticker,
        timestamp_utc=datetime.now(timezone.utc),
        predicted_close=predicted_close,
        confidence_pct=max(0.0, min(100.0, 100.0 - drift_score * 100)),
        prediction_lower=predicted_close - std,
        prediction_upper=predicted_close + std,
        model_version=_model_registry.current_version,
        model_name="gradient_boosting",
        feature_version="v2.1.0",
        schema_hash=_SCHEMA_HASH,
        drift_score=drift_score,
    )


@app.get("/v1/tickers")
async def list_tickers() -> dict[str, list[str]]:
    return {"tickers": []}

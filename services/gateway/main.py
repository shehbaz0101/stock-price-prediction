"""API Gateway — proxies to downstream services."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .health import get_health
from .http_client import close_http, _get, _post, _delete
from .middleware import AuthMiddleware, RateLimitMiddleware

INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion:8001")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8002")
LLM_URL       = os.getenv("LLM_URL",       "http://llm-insight:8003")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_http()


app = FastAPI(title="Stock Platform Gateway", lifespan=lifespan)
app.add_middleware(RateLimitMiddleware, redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/1"))


@app.get("/health")
async def health() -> dict[str, Any]:
    return await get_health()


@app.get("/v1/health")
async def v1_health() -> dict[str, Any]:
    return await get_health()


@app.get("/v1/tickers")
async def list_tickers() -> dict[str, Any]:
    return await _get(f"{INGESTION_URL}/v1/tickers")


@app.post("/v1/tickers/{ticker}")
async def add_ticker(ticker: str) -> dict[str, Any]:
    return await _post(f"{INGESTION_URL}/v1/tickers/{ticker}", {})


@app.delete("/v1/tickers/{ticker}")
async def remove_ticker(ticker: str) -> dict[str, Any]:
    return await _delete(f"{INGESTION_URL}/v1/tickers/{ticker}")


@app.get("/v1/predict/{ticker}")
async def get_prediction(ticker: str) -> dict[str, Any]:
    return await _get(f"{INFERENCE_URL}/v1/predict/{ticker}")


@app.get("/v1/model/info")
async def model_info() -> dict[str, Any]:
    return await _get(f"{INFERENCE_URL}/v1/model/info")


@app.post("/v1/insight")
async def get_insight(body: dict[str, Any]) -> dict[str, Any]:
    return await _post(f"{LLM_URL}/v1/insight", body)

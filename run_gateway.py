"""
API Gateway - port 8000
Single entry point. Proxies to ingestion (8001), inference (8002), llm_insight (8003).
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_env_pp = os.environ.get("PYTHONPATH", "")
for _p in _env_pp.split(os.pathsep):
    _p = os.path.abspath(_p)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("gateway")

INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8001")
AGENT_URL     = os.getenv("AGENT_URL",     "http://localhost:8004")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8002")
LLM_URL       = os.getenv("LLM_URL",       "http://localhost:8003")

_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=15.0)
    log.info("Gateway ready on port 8000")
    yield
    if _client:
        await _client.aclose()


app = FastAPI(title="Stock Platform Gateway", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _proxy_get(url: str) -> Any:
    try:
        r = await _http().get(url)
        return r.json()
    except Exception as exc:
        return {"error": str(exc), "url": url}


async def _proxy_post(url: str, body: dict[str, Any]) -> Any:
    try:
        r = await _http().post(url, json=body)
        return r.json()
    except Exception as exc:
        return {"error": str(exc), "url": url}


async def _proxy_delete(url: str) -> Any:
    try:
        r = await _http().delete(url)
        return r.json()
    except Exception as exc:
        return {"error": str(exc), "url": url}


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Stock Platform Gateway", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, Any]:
    import asyncio
    ing, inf, llm = await asyncio.gather(
        _proxy_get(f"{INGESTION_URL}/health"),
        _proxy_get(f"{INFERENCE_URL}/health"),
        _proxy_get(f"{LLM_URL}/health"),
    )
    all_ok = all(
        isinstance(s, dict) and s.get("status") == "ok"
        for s in [ing, inf, llm]
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "services": {
            "ingestion":   ing.get("status", "error") if isinstance(ing, dict) else "error",
            "inference":   inf.get("status", "error") if isinstance(inf, dict) else "error",
            "llm_insight": llm.get("status", "error") if isinstance(llm, dict) else "error",
        },
    }


@app.get("/v1/health")
async def v1_health() -> dict[str, Any]:
    return await health()


@app.get("/v1/tickers")
async def list_tickers() -> Any:
    return await _proxy_get(f"{INGESTION_URL}/v1/tickers")


@app.post("/v1/tickers/{ticker}")
async def add_ticker(ticker: str) -> Any:
    return await _proxy_post(f"{INGESTION_URL}/v1/tickers/{ticker}", {})


@app.delete("/v1/tickers/{ticker}")
async def remove_ticker(ticker: str) -> Any:
    return await _proxy_delete(f"{INGESTION_URL}/v1/tickers/{ticker}")


@app.post("/v1/ingest/now")
async def ingest_now(request: Request) -> Any:
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}
    return await _proxy_post(f"{INGESTION_URL}/v1/ingest/now", body)


@app.get("/v1/predict/{ticker}")
async def get_prediction(ticker: str) -> Any:
    return await _proxy_get(f"{INFERENCE_URL}/v1/predict/{ticker}")


@app.get("/v1/model/info")
async def model_info() -> Any:
    return await _proxy_get(f"{INFERENCE_URL}/v1/model/info")


@app.post("/v1/insight")
async def get_insight(request: Request) -> Any:
    body: dict[str, Any] = await request.json()
    return await _proxy_post(f"{LLM_URL}/v1/insight", body)


@app.get("/v1/stats")
async def stats() -> Any:
    return await _proxy_get(f"{INGESTION_URL}/v1/stats")


@app.get("/v1/agent/health")
async def agent_health() -> Any:
    return await _proxy_get(f"{AGENT_URL}/health")


@app.post("/v1/agent/analyse")
async def agent_analyse(request: Request) -> Any:
    """Stream SSE from agent service through gateway."""
    import httpx
    body = await request.body()
    async def stream_gen():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{AGENT_URL}/v1/agent/analyse",
                content=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/v1/llm/health")
async def llm_health() -> Any:
    return await _proxy_get(f"{LLM_URL}/health")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
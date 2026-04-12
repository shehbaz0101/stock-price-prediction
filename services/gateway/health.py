"""Health aggregator. Uses _get_http() — never calls on None."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from .http_client import _get_http

INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion:8001")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8002")
LLM_URL       = os.getenv("LLM_URL",       "http://llm-insight:8003")

_HEALTH_URLS: dict[str, str] = {
    "ingestion":   f"{INGESTION_URL}/health",
    "inference":   f"{INFERENCE_URL}/health",
    "llm_insight": f"{LLM_URL}/health",
}


async def _check(name: str, url: str) -> tuple[str, dict[str, Any]]:
    """FIX: uses _get_http() — guaranteed non-None client."""
    try:
        resp = await _get_http().get(url, timeout=3.0)
        return name, {"status": "ok" if resp.status_code == 200 else "degraded"}
    except Exception:
        return name, {"status": "unreachable"}


async def get_health() -> dict[str, Any]:
    checks   = await asyncio.gather(*[_check(n, u) for n, u in _HEALTH_URLS.items()])
    statuses: dict[str, dict[str, Any]] = dict(checks)
    overall  = "ok" if all(v["status"] == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses, "ts": time.time()}

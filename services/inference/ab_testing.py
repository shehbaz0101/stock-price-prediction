"""A/B test router. Renamed lazy-init method to _get_redis_conn to avoid attribute shadowing."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url

log = logging.getLogger(__name__)


class ABTestRouter:
    """Deterministic routing: same (ticker, date) always maps to same variant."""

    _CONFIG_KEY    = "ab:config"
    _METRICS_KEY   = "ab:metrics:{variant}"
    _OUTCOMES_KEY  = "ab:outcomes:{ticker}:{variant}"

    def __init__(self, redis_url: str = "redis://localhost:6379/2") -> None:
        self._redis_url = redis_url
        # FIX: attribute is _redis_conn, not _redis, to avoid shadowing
        self._redis_conn: Redis | None = None  # type: ignore[type-arg]
        self._champion_model:   Any | None = None
        self._challenger_model: Any | None = None

    # FIX: renamed from _redis() (conflicted with self._redis attribute) to _get_redis_conn()
    async def _get_redis_conn(self) -> Redis:  # type: ignore[type-arg]
        if self._redis_conn is None:
            self._redis_conn = await redis_from_url(
                self._redis_url, decode_responses=True
            )
        return self._redis_conn

    async def configure(
        self,
        champion_model: Any,
        challenger_model: Any,
        challenger_pct: float = 0.1,
    ) -> None:
        self._champion_model    = champion_model
        self._challenger_model  = challenger_model
        r = await self._get_redis_conn()
        await r.set(
            self._CONFIG_KEY,
            json.dumps({"challenger_pct": challenger_pct, "active": True}),
        )

    def _route(self, ticker: str, date_str: str, challenger_pct: float = 0.1) -> str:
        digest   = hashlib.sha256(f"{ticker}:{date_str}".encode()).digest()
        fraction = digest[0] / 255.0
        return "challenger" if fraction < challenger_pct else "champion"

    async def get_config(self) -> dict[str, Any]:
        r = await self._get_redis_conn()
        raw = await r.get(self._CONFIG_KEY)
        return json.loads(raw) if raw else {}

    async def record_outcome(
        self, ticker: str, date_str: str, variant: str, outcome: float
    ) -> None:
        r = await self._get_redis_conn()
        key = self._OUTCOMES_KEY.format(ticker=ticker, variant=variant)
        await r.rpush(key, str(outcome))   # type: ignore[arg-type]
        await r.expire(key, 86400 * 30)

    async def close(self) -> None:
        if self._redis_conn is not None:
            await self._redis_conn.aclose()
            self._redis_conn = None

"""
Feature store backed by Redis.
Falls back to fakeredis.FakeAsyncRedis (in-memory, no server needed) when:
  - USE_FAKE_REDIS=1 is set in environment, OR
  - Redis server is unreachable on first connection attempt.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_ONLINE_KEY_PREFIX  = "fv:online:"
_HISTORY_KEY_PREFIX = "fv:history:"


def _make_fake_client() -> Any:
    """Return a fakeredis async client. fakeredis.FakeAsyncRedis lives
    directly on the top-level fakeredis package - no submodule needed."""
    import fakeredis  # type: ignore[import]
    client: Any = fakeredis.FakeAsyncRedis(decode_responses=True)
    log.warning(
        "Using fakeredis in-memory store "
        "(USE_FAKE_REDIS=1 or Redis unreachable). "
        "Data resets on service restart."
    )
    return client


async def _create_client(redis_url: str) -> Any:
    """
    Create and return a working async Redis-compatible client.
    Tries real Redis first; falls back to FakeAsyncRedis on any error.
    """
    use_fake = os.environ.get("USE_FAKE_REDIS", "0").strip() == "1"

    if not use_fake:
        try:
            from redis.asyncio import from_url as redis_from_url
            client: Any = await redis_from_url(redis_url, decode_responses=True)
            # Verify the connection is alive.
            # ping() IS a coroutine at runtime even though stubs say bool.
            await client.ping()  # type: ignore[misc]
            log.info("Connected to Redis at %s", redis_url)
            return client
        except Exception as exc:
            log.warning("Redis unavailable (%s) - falling back to fakeredis", exc)
            os.environ["USE_FAKE_REDIS"] = "1"

    return _make_fake_client()


class FeatureStoreClient:
    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._conn: Any | None = None

    async def _get_conn(self) -> Any:
        """Return live client, creating it on first call."""
        if self._conn is None:
            self._conn = await _create_client(self._redis_url)
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.aclose()
            except Exception:
                pass
            self._conn = None

    async def store_online(
        self, ticker: str, features: Any, ttl: int = 3600
    ) -> None:
        conn = await self._get_conn()
        payload: str = (
            features.model_dump_json()
            if hasattr(features, "model_dump_json")
            else json.dumps(features)
        )
        key      = f"{_ONLINE_KEY_PREFIX}{ticker.upper()}"
        hist_key = f"{_HISTORY_KEY_PREFIX}{ticker.upper()}"
        await conn.set(key, payload, ex=ttl)
        await conn.lpush(hist_key, payload)   # type: ignore[arg-type]
        await conn.ltrim(hist_key, 0, 999)    # type: ignore[arg-type]
        log.info("Feature vector stored  ticker=%s", ticker)

    async def get_online(self, ticker: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        raw: str | None = await conn.get(
            f"{_ONLINE_KEY_PREFIX}{ticker.upper()}"
        )
        if raw is None:
            log.warning("Feature cache miss  ticker=%s", ticker)
            return None
        return json.loads(raw)  # type: ignore[no-any-return]

    async def get_history(
        self, ticker: str, n: int = 100
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        raw_list: list[str] = await conn.lrange(  # type: ignore[assignment]
            f"{_HISTORY_KEY_PREFIX}{ticker.upper()}", 0, n - 1
        )
        return [json.loads(r) for r in raw_list]
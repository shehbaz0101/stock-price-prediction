"""Auth + rate-limit middleware. Guards request.client for None."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Sequence

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url

log = logging.getLogger(__name__)

_VALID_KEY_HASHES: set[str] = set()

_BYPASS: frozenset[str] = frozenset({
    "/health", "/v1/health", "/metrics", "/docs", "/openapi.json", "/redoc",
})


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _client_host(request: Request) -> str:
    """FIX: request.client is Optional — guard before accessing .host."""
    return request.client.host if request.client else "unknown"


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, valid_key_hashes: Sequence[str] = ()) -> None:
        super().__init__(app)
        _VALID_KEY_HASHES.update(valid_key_hashes)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _BYPASS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            log.warning("Missing API key",
                        extra={"path": request.url.path, "client": _client_host(request)})
            return JSONResponse(status_code=401,
                                content={"error": "Missing X-API-Key header"})

        if _hash_key(api_key) not in _VALID_KEY_HASHES:
            log.warning("Invalid API key",
                        extra={"path": request.url.path, "client": _client_host(request)})
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter backed by Redis."""

    def __init__(
        self,
        app: ASGIApp,
        redis_url: str = "redis://localhost:6379/1",
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._redis_url      = redis_url
        self._max_requests   = max_requests
        self._window_seconds = window_seconds
        self._redis_conn: Redis | None = None  # type: ignore[type-arg]

    async def _get_redis(self) -> Redis:  # type: ignore[type-arg]
        if self._redis_conn is None:
            self._redis_conn = await redis_from_url(
                self._redis_url, decode_responses=True
            )
        return self._redis_conn

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _BYPASS:
            return await call_next(request)

        api_key   = request.headers.get("X-API-Key", "")
        client_id = _hash_key(api_key)[:16] if api_key else _client_host(request)
        redis_key = f"ratelimit:{client_id}"
        now       = time.time()
        window_start = now - self._window_seconds

        try:
            r = await self._get_redis()
            pipe = r.pipeline()
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            pipe.zadd(redis_key, {str(now): now})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, self._window_seconds * 2)
            results = await pipe.execute()
            request_count: int = int(results[2])
        except Exception as exc:
            log.error("Rate-limit Redis error", extra={"exc": str(exc)})
            return await call_next(request)

        if request_count > self._max_requests:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": str(self._window_seconds)},
            )
        return await call_next(request)

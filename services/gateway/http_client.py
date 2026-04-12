"""Shared async HTTP client. Lazy-init so it is never None when called."""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

_http: httpx.AsyncClient | None = None


def _get_http() -> httpx.AsyncClient:
    """Return the shared client, creating it lazily."""
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http


async def close_http() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


async def _get(url: str, *, timeout: float | None = None) -> dict[str, Any]:
    try:
        kw: dict[str, Any] = {}
        if timeout is not None:
            kw["timeout"] = timeout
        resp = await _get_http().get(url, **kw)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Upstream unavailable: {exc}") from exc


async def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = await _get_http().post(url, json=body)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, detail=str(exc)) from exc


async def _delete(url: str) -> dict[str, Any]:
    try:
        resp = await _get_http().delete(url)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

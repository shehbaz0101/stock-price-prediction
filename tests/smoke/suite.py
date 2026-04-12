"""Smoke test suite. Uses async context manager — _client never None."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


class SmokeTestSuite:
    def __init__(self, base_url: str, ticker: str = "RELIANCE.NS") -> None:
        self.base_url = base_url.rstrip("/")
        self.ticker   = ticker
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SmokeTestSuite":
        self._client = httpx.AsyncClient(timeout=15.0)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _c(self) -> httpx.AsyncClient:
        """Return live client — raises if called outside context manager."""
        if self._client is None:
            raise RuntimeError("Use `async with SmokeTestSuite(...):`")
        return self._client

    # --- individual tests ------------------------------------------------

    async def _test_health(self) -> TestResult:
        r = await self._c().get(f"{self.base_url}/v1/health")
        data: dict[str, Any] = r.json()
        return TestResult("_test_health", r.status_code == 200,
                          data.get("status", "unknown"))

    async def _test_all_services_healthy(self) -> TestResult:
        r    = await self._c().get(f"{self.base_url}/v1/health")
        data = r.json()
        svcs: dict[str, Any] = data.get("services", {})
        down = [s for s, info in svcs.items() if info.get("status") != "ok"]
        return TestResult("_test_all_services_healthy", not down,
                          f"Down: {down}" if down else f"All {len(svcs)} services healthy")

    async def _test_model_info(self) -> TestResult:
        r    = await self._c().get(f"{self.base_url}/v1/model/info")
        data = r.json()
        ok   = "version" in data and "feature_hash" in data
        return TestResult("_test_model_info", ok,
                          f"v{data.get('version')} hash={str(data.get('feature_hash','?'))[:8]}")

    async def _test_ticker_list(self) -> TestResult:
        r       = await self._c().get(f"{self.base_url}/v1/tickers")
        tickers: list[str] = r.json().get("tickers", [])
        return TestResult("_test_ticker_list", len(tickers) > 0,
                          f"{len(tickers)} active tickers")

    async def _test_prediction(self) -> TestResult:
        r = await self._c().get(f"{self.base_url}/v1/predict/{self.ticker}")
        return TestResult("_test_prediction", r.status_code == 200,
                          f"HTTP {r.status_code}")

    async def _test_prediction_cache(self) -> TestResult:
        t0 = time.perf_counter()
        await self._c().get(f"{self.base_url}/v1/predict/{self.ticker}")
        first  = time.perf_counter() - t0
        t0 = time.perf_counter()
        await self._c().get(f"{self.base_url}/v1/predict/{self.ticker}")
        second = time.perf_counter() - t0
        ok = second < first * 2.0
        return TestResult("_test_prediction_cache", ok,
                          f"1st={first*1000:.0f}ms 2nd={second*1000:.0f}ms")

    async def _test_ticker_lifecycle(self) -> TestResult:
        test_ticker = "SMOKE_TEST_XYZ"
        r_add = await self._c().post(f"{self.base_url}/v1/tickers/{test_ticker}")
        if r_add.status_code not in (200, 201, 409):
            return TestResult("_test_ticker_lifecycle", False,
                              f"Add failed: HTTP {r_add.status_code}")
        r_del = await self._c().delete(f"{self.base_url}/v1/tickers/{test_ticker}")
        return TestResult("_test_ticker_lifecycle",
                          r_del.status_code in (200, 204, 404), "Add+Remove OK")

    async def _test_latency(self) -> TestResult:
        times: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            await self._c().get(f"{self.base_url}/v1/predict/{self.ticker}")
            times.append((time.perf_counter() - t0) * 1000)
        p50 = sorted(times)[len(times) // 2]
        return TestResult("_test_latency", p50 < 500,
                          f"p50={p50:.0f}ms min={min(times):.0f}ms max={max(times):.0f}ms")

    async def _test_metrics(self) -> TestResult:
        try:
            r  = await self._c().get(f"{self.base_url}/metrics", timeout=5.0)
            ok = r.status_code == 200 and "http_requests_total" in r.text
            return TestResult("_test_metrics", ok, f"HTTP {r.status_code}")
        except httpx.RequestError as exc:
            return TestResult("_test_metrics", False, str(exc))

    # --- runner ----------------------------------------------------------

    async def run_all(self) -> list[TestResult]:
        results: list[TestResult] = []
        for fn in [
            self._test_health,
            self._test_all_services_healthy,
            self._test_model_info,
            self._test_ticker_list,
            self._test_prediction,
            self._test_prediction_cache,
            self._test_ticker_lifecycle,
            self._test_latency,
            self._test_metrics,
        ]:
            try:
                results.append(await fn())
            except Exception as exc:
                results.append(TestResult(fn.__name__, False, f"Exception: {exc}"))
        return results

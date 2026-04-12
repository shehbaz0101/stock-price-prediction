"""Market data adapters. Factory accepts ticker= keyword on all adapters."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from shared.contracts import DataSource

log = logging.getLogger(__name__)


class BaseMarketAdapter:
    """All adapters must accept ticker as a keyword argument."""
    source: DataSource = DataSource.YFINANCE

    def __init__(self, ticker: str, **_kwargs: Any) -> None:
        self.ticker = ticker.upper()

    async def fetch_latest(self) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceAdapter(BaseMarketAdapter):
    source = DataSource.YFINANCE

    def __init__(self, ticker: str, period: str = "5d", interval: str = "1d") -> None:
        super().__init__(ticker)
        self.period = period
        self.interval = interval

    async def fetch_latest(self) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_sync)

    def _fetch_sync(self) -> pd.DataFrame:
        try:
            df = yf.download(
                self.ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return pd.DataFrame()
            return pd.DataFrame(df)
        except Exception as exc:
            log.error("YFinance fetch failed", extra={"ticker": self.ticker, "exc": str(exc)})
            return pd.DataFrame()


class PolygonAdapter(BaseMarketAdapter):
    source = DataSource.POLYGON

    def __init__(self, ticker: str, api_key: str = "") -> None:
        super().__init__(ticker)
        self.api_key = api_key

    async def fetch_latest(self) -> pd.DataFrame:
        raise NotImplementedError("Polygon adapter not implemented")


_ADAPTERS: dict[DataSource, type[BaseMarketAdapter]] = {
    DataSource.YFINANCE: YFinanceAdapter,
    DataSource.POLYGON: PolygonAdapter,
}


def MarketDataFetcher(
    ticker: str,
    source: DataSource = DataSource.YFINANCE,
    **kwargs: Any,
) -> BaseMarketAdapter:
    """Factory — all adapters accept ticker= because they inherit BaseMarketAdapter."""
    cls = _ADAPTERS.get(source)
    if cls is None:
        raise ValueError(f"Unsupported data source: {source}")
    return cls(ticker=ticker, **kwargs)

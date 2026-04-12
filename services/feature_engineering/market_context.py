"""Market context enricher. _fetch is a sync def — called via run_in_executor."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

INDICES: dict[str, str] = {
    "nifty50": "^NSEI",
    "sp500":   "^GSPC",
    "vix":     "^VIX",
}


class MarketContextFeatures:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class MarketContextEnricher:
    async def _compute(
        self, ticker: str, as_of: datetime
    ) -> MarketContextFeatures:
        loop = asyncio.get_event_loop()

        # FIX: _fetch MUST be a plain def (sync) — run_in_executor cannot take async.
        # Return type is always pd.DataFrame, never None.
        def _fetch(t: str, days: int) -> pd.DataFrame:
            end   = as_of
            start = end - timedelta(days=days)
            try:
                df = yf.download(
                    t, start=start, end=end,
                    auto_adjust=True, progress=False,
                )
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return pd.DataFrame()
                return pd.DataFrame(df)
            except Exception as exc:
                log.warning("yfinance fetch failed",
                            extra={"ticker": t, "exc": str(exc)})
                return pd.DataFrame()

        nifty_df, sp500_df, vix_df, ticker_df = await asyncio.gather(
            loop.run_in_executor(None, _fetch, INDICES["nifty50"], 30),
            loop.run_in_executor(None, _fetch, INDICES["sp500"],   30),
            loop.run_in_executor(None, _fetch, INDICES["vix"],     30),
            loop.run_in_executor(None, _fetch, ticker,             30),
        )

        def _last_ret(df: pd.DataFrame) -> float:
            if df.empty or "Close" not in df.columns:
                return 0.0
            c = df["Close"].dropna()
            return float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) if len(c) >= 2 else 0.0

        def _last_val(df: pd.DataFrame) -> float:
            if df.empty or "Close" not in df.columns:
                return 0.0
            return float(df["Close"].dropna().iloc[-1])

        return MarketContextFeatures(data={
            "nifty50_return_1d": _last_ret(nifty_df),
            "sp500_return_1d":   _last_ret(sp500_df),
            "vix_last":          _last_val(vix_df),
            "ticker_return_1d":  _last_ret(ticker_df),
            "computed_at":       time.time(),
        })

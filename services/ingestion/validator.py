"""Tick validator — converts raw dict payloads into OHLCVTick."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from shared.contracts import DataSource, OHLCVTick


class TickValidator:
    def validate(self, raw: dict[str, Any], ticker: str) -> OHLCVTick:
        ts_raw = raw.get("Datetime") or raw.get("timestamp_utc") or raw.get("Date")
        if ts_raw is None:
            raise ValueError(f"Missing timestamp in payload for {ticker}")

        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        close = float(raw.get("Close") or raw.get("close") or 0)
        open_ = float(raw.get("Open") or raw.get("open") or close)
        high  = float(raw.get("High") or raw.get("high") or close)
        low   = float(raw.get("Low") or raw.get("low") or close)
        vol   = float(raw.get("Volume") or raw.get("volume") or 0)

        # Fix OHLCV so high >= open/close >= low
        high  = max(high, open_, close)
        low   = min(low, open_, close)

        return OHLCVTick(
            ticker=ticker,
            timestamp_utc=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=vol,
            source=DataSource(raw.get("source", DataSource.YFINANCE)),
        )

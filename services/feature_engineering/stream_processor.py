"""Real-time feature computation from a rolling buffer of OHLCV ticks."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from shared.contracts import FEATURE_KEYS, FeatureVector, OHLCVTick, _SCHEMA_HASH

log = logging.getLogger(__name__)


class TickerStateBuffer:
    """Maintains a rolling window of ticks for a single ticker."""

    def __init__(self, ticker: str, max_size: int = 200) -> None:
        self.ticker = ticker.upper()
        self.max_size = max_size
        self._buf: deque[OHLCVTick] = deque(maxlen=max_size)

    def push(self, tick: OHLCVTick) -> bool:
        if tick.ticker != self.ticker:
            return False
        self._buf.append(tick)
        return True

    @property
    def ready(self) -> bool:
        return len(self._buf) >= 20

    @property
    def ticks(self) -> list[OHLCVTick]:
        return list(self._buf)


class RealTimeFeatureComputer:
    """Compute features from a TickerStateBuffer."""

    def compute(self, buf: TickerStateBuffer, tick: OHLCVTick) -> FeatureVector:
        ticks = buf.ticks
        closes = np.array([t.close for t in ticks], dtype=float)
        volumes = np.array([t.volume for t in ticks], dtype=float)

        features: dict[str, float] = {}

        # Returns
        features["close"] = float(closes[-1])
        features["returns_1d"] = _pct_change(closes, 1)
        features["returns_5d"] = _pct_change(closes, 5)
        features["returns_20d"] = _pct_change(closes, 20)

        # Volume ratio
        mean_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        features["volume_ratio"] = float(volumes[-1]) / (mean_vol or 1.0)

        # RSI-14
        features["rsi_14"] = _rsi(closes, 14)

        # MACD
        macd, macd_signal = _macd(closes)
        features["macd"] = macd
        features["macd_signal"] = macd_signal

        # Bollinger bands
        bb_upper, bb_lower = _bollinger(closes, 20, 2.0)
        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower

        # ATR-14
        highs  = np.array([t.high for t in ticks], dtype=float)
        lows   = np.array([t.low for t in ticks], dtype=float)
        features["atr_14"] = _atr(highs, lows, closes, 14)

        # OBV
        features["obv"] = _obv(closes, volumes)

        # Market context (filled in later by MarketContextEnricher)
        for k in ["nifty50_return_1d", "sp500_return_1d", "vix_last", "ticker_return_1d"]:
            features[k] = 0.0

        return FeatureVector(
            ticker=tick.ticker,
            timestamp_utc=tick.timestamp_utc,
            schema_hash=_SCHEMA_HASH,
            features=features,
        )


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _pct_change(arr: np.ndarray, n: int) -> float:
    if len(arr) <= n:
        return 0.0
    prev = arr[-(n + 1)]
    curr = arr[-1]
    return float((curr - prev) / (prev or 1.0))


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _macd(closes: np.ndarray) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    return float(macd_line[-1]), float(signal[-1])


def _bollinger(closes: np.ndarray, period: int = 20, n_std: float = 2.0) -> tuple[float, float]:
    if len(closes) < period:
        return float(closes[-1]) * 1.02, float(closes[-1]) * 0.98
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid + n_std * std, mid - n_std * std


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    n = min(period, len(closes) - 1)
    trs: list[float] = []
    for i in range(-n, 0):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs))


def _obv(closes: np.ndarray, volumes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.0
    obv_val = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_val += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_val -= volumes[i]
    return obv_val

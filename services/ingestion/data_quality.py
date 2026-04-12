"""Data quality checks. DataQualityReport(ticker=...) — not tick=."""
from __future__ import annotations

import logging
import statistics
from collections import deque
from datetime import datetime

from shared.contracts import DataQualityReport, OHLCVTick, QualityCheck

log = logging.getLogger(__name__)


class DataQualityMonitor:
    MAX_WINDOW = 200

    def __init__(self) -> None:
        self._close_windows: dict[str, deque[float]] = {}
        self._volume_windows: dict[str, deque[float]] = {}
        self._last_ts: dict[str, datetime] = {}

    def check(self, tick: OHLCVTick) -> DataQualityReport:
        """Run all quality checks. Returns a DataQualityReport."""
        # FIX: DataQualityReport takes ticker: str, not tick: OHLCVTick
        report = DataQualityReport(ticker=tick.ticker)
        ticker = tick.ticker

        if ticker not in self._close_windows:
            self._close_windows[ticker] = deque(maxlen=self.MAX_WINDOW)
            self._volume_windows[ticker] = deque(maxlen=self.MAX_WINDOW)

        report.checks.append(self._check_price_floor(tick))
        report.checks.append(self._check_spread(tick))

        if len(self._close_windows[ticker]) >= 5:
            report.checks.append(self._check_price_spike(tick))

        if len(self._volume_windows[ticker]) >= 5:
            report.checks.append(self._check_volume_anomaly(tick))

        if ticker in self._last_ts:
            report.checks.append(self._check_ts_monotonic(tick))

        self._close_windows[ticker].append(tick.close)
        self._volume_windows[ticker].append(tick.volume)
        self._last_ts[ticker] = tick.timestamp_utc
        return report

    def _check_price_floor(self, tick: OHLCVTick) -> QualityCheck:
        passed = tick.close > 0 and tick.open > 0
        return QualityCheck(name="price_floor", passed=passed,
                            detail="" if passed else "Zero or negative price")

    def _check_spread(self, tick: OHLCVTick) -> QualityCheck:
        passed = tick.high >= max(tick.open, tick.close) >= tick.low
        return QualityCheck(name="ohlcv_spread", passed=passed,
                            detail="" if passed else "OHLCV inconsistency")

    def _check_price_spike(self, tick: OHLCVTick) -> QualityCheck:
        window = list(self._close_windows[tick.ticker])
        median = statistics.median(window)
        ratio = abs(tick.close - median) / (median or 1.0)
        passed = ratio < 0.20
        return QualityCheck(name="price_spike", passed=passed,
                            detail="" if passed else f"Spike {ratio:.1%} from median {median:.2f}")

    def _check_volume_anomaly(self, tick: OHLCVTick) -> QualityCheck:
        window = list(self._volume_windows[tick.ticker])
        mean_vol = statistics.mean(window) or 1.0
        ratio = tick.volume / mean_vol
        passed = 0.01 < ratio < 50.0
        return QualityCheck(name="volume_anomaly", passed=passed,
                            detail="" if passed else f"Volume ratio {ratio:.1f}x vs mean")

    def _check_ts_monotonic(self, tick: OHLCVTick) -> QualityCheck:
        last = self._last_ts[tick.ticker]
        passed = tick.timestamp_utc > last
        return QualityCheck(name="ts_monotonic", passed=passed,
                            detail="" if passed else f"Non-monotonic: {tick.timestamp_utc} <= {last}")

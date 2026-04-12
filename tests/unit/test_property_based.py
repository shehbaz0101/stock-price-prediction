"""Property-based tests. hypothesis installed via requirements.txt."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from shared.contracts import (
    DataSource,
    FeatureVector,
    OHLCVTick,
    _SCHEMA_HASH,
    FEATURE_KEYS,
)
from services.feature_engineering.stream_processor import (
    RealTimeFeatureComputer,
    TickerStateBuffer,
)
from services.ingestion.data_quality import DataQualityMonitor
from services.ingestion.validator import TickValidator

# Run with:
#   pytest tests/unit/test_property_based.py -v --hypothesis-show-statistics

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_TICKERS = st.sampled_from(["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"])

_PRICE = st.floats(min_value=10.0, max_value=100_000.0,
                   allow_nan=False, allow_infinity=False)

_VOLUME = st.floats(min_value=1.0, max_value=1e9,
                    allow_nan=False, allow_infinity=False)

_DATE_OFFSET = st.integers(min_value=0, max_value=3000)


@st.composite
def valid_tick(draw: st.DrawFn) -> OHLCVTick:
    ticker = draw(_TICKERS)
    close  = draw(_PRICE)
    spread = draw(st.floats(min_value=0.0, max_value=close * 0.10,
                            allow_nan=False, allow_infinity=False))
    high   = close + spread
    low    = max(0.01, close - spread)
    open_  = draw(st.floats(min_value=low, max_value=high,
                            allow_nan=False, allow_infinity=False))
    volume = draw(_VOLUME)
    offset = draw(_DATE_OFFSET)
    ts     = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset)
    return OHLCVTick(
        ticker=ticker, timestamp_utc=ts,
        open=open_, high=high, low=low, close=close, volume=volume,
        source=DataSource.YFINANCE,
    )


@st.composite
def tick_sequence(draw: st.DrawFn, min_size: int = 25) -> list[OHLCVTick]:
    ticker = draw(_TICKERS)
    n      = draw(st.integers(min_value=min_size, max_value=60))
    ticks: list[OHLCVTick] = []
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        close  = draw(st.floats(min_value=100.0, max_value=50_000.0,
                                allow_nan=False, allow_infinity=False))
        spread = draw(st.floats(min_value=0.0, max_value=close * 0.05,
                                allow_nan=False, allow_infinity=False))
        high   = close + spread
        low    = max(0.01, close - spread)
        ticks.append(OHLCVTick(
            ticker=ticker,
            timestamp_utc=base_ts + timedelta(days=i),
            open=close, high=high, low=low, close=close,
            volume=float(draw(st.integers(min_value=1000, max_value=10_000_000))),
        ))
    return ticks


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestTickValidatorProperties:
    @given(tick=valid_tick())
    @settings(max_examples=200)
    def test_valid_tick_round_trips(self, tick: OHLCVTick) -> None:
        """A tick built from valid data should pass the validator."""
        validator = TickValidator()
        raw = {
            "Datetime": tick.timestamp_utc.isoformat(),
            "Open":     str(tick.open),
            "High":     str(tick.high),
            "Low":      str(tick.low),
            "Close":    str(tick.close),
            "Volume":   str(tick.volume),
        }
        validated = validator.validate(raw, tick.ticker)
        assert validated.ticker == tick.ticker.upper()
        assert validated.close  > 0
        assert validated.high   >= validated.close >= validated.low

    @given(close=_PRICE)
    def test_negative_price_rejected_after_normalisation(self, close: float) -> None:
        """The OHLCVTick model rejects close <= 0."""
        assume(close > 0)
        tick = OHLCVTick(
            ticker="TEST.NS",
            timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            open=close, high=close, low=close, close=close, volume=1000.0,
        )
        assert tick.close > 0


class TestDataQualityProperties:
    @given(ticks=tick_sequence(min_size=6))
    @settings(max_examples=100)
    def test_check_never_raises(self, ticks: list[OHLCVTick]) -> None:
        """DataQualityMonitor.check() must never throw."""
        monitor = DataQualityMonitor()
        for tick in ticks:
            report = monitor.check(tick)
            assert report.ticker == tick.ticker.upper()


class TestFeatureComputerProperties:
    @given(ticks=tick_sequence(min_size=25))
    @settings(max_examples=50)
    def test_feature_keys_present(self, ticks: list[OHLCVTick]) -> None:
        """Once the buffer is ready, all FEATURE_KEYS must appear in features."""
        ticker = ticks[0].ticker
        buf    = TickerStateBuffer(ticker, max_size=200)
        comp   = RealTimeFeatureComputer()
        fv     = None
        for tick in ticks:
            buf.push(tick)
            if buf.ready:
                fv = comp.compute(buf, tick)
        if fv is not None:
            for k in FEATURE_KEYS:
                assert k in fv.features, f"Missing feature: {k}"

    @given(ticks=tick_sequence(min_size=25))
    @settings(max_examples=50)
    def test_features_are_finite(self, ticks: list[OHLCVTick]) -> None:
        """All computed feature values must be finite (no NaN/Inf)."""
        ticker = ticks[0].ticker
        buf    = TickerStateBuffer(ticker, max_size=200)
        comp   = RealTimeFeatureComputer()
        for tick in ticks:
            buf.push(tick)
            if buf.ready:
                fv = comp.compute(buf, tick)
                for name, val in fv.features.items():
                    assert np.isfinite(val), f"Non-finite feature {name}={val}"

    @given(ticks=tick_sequence(min_size=25))
    @settings(max_examples=50)
    def test_rsi_in_range(self, ticks: list[OHLCVTick]) -> None:
        """RSI must always be in [0, 100]."""
        ticker = ticks[0].ticker
        buf    = TickerStateBuffer(ticker, max_size=200)
        comp   = RealTimeFeatureComputer()
        for tick in ticks:
            buf.push(tick)
            if buf.ready:
                fv = comp.compute(buf, tick)
                rsi = fv.features.get("rsi_14", 50.0)
                assert 0.0 <= rsi <= 100.0, f"RSI out of range: {rsi}"


class TestDriftMonitorProperties:
    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=10, max_size=200,
        )
    )
    def test_psi_non_negative(self, values: list[float]) -> None:
        """PSI score must always be >= 0."""
        from services.inference.drift_monitor import DriftMonitor
        monitor = DriftMonitor()
        baseline = {"feat": values}
        monitor._baseline = baseline
        current  = {"feat": values[-1]}
        score    = monitor.score(current)
        assert score >= 0.0

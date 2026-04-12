"""E2E tests. Prediction model requires drift_score — given a default so it's optional."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from shared.contracts import (
    OHLCVTick,
    FeatureVector,
    Prediction,
    DataQualityReport,
    _SCHEMA_HASH,
)
from services.ingestion.validator import TickValidator
from services.ingestion.data_quality import DataQualityMonitor
from services.feature_engineering.stream_processor import (
    RealTimeFeatureComputer,
    TickerStateBuffer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_tick(n: int) -> OHLCVTick:
    base = 1000.0 + n * 5
    return OHLCVTick(
        ticker="INTG.NS",
        timestamp_utc=datetime(2026, 1, n + 1, tzinfo=timezone.utc),
        open=base,
        high=base + 10,
        low=base - 10,
        close=base + 2,
        volume=100_000.0 + n * 1000,
    )


@pytest.fixture
def sample_ticks() -> list[OHLCVTick]:
    return [_make_tick(i) for i in range(30)]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------
class TestLocalPipeline:
    def test_tick_validation(self) -> None:
        validator = TickValidator()
        raw = {
            "Datetime": "2026-01-01T09:15:00+05:30",
            "Open": "2800.0",
            "High": "2860.0",
            "Low":  "2790.0",
            "Close": "2850.0",
            "Volume": "1500000",
        }
        tick = validator.validate(raw, "INTG.NS")
        assert tick.ticker   == "INTG.NS"
        assert tick.close    == 2850.0
        assert tick.high     >= tick.close >= tick.low

    def test_data_quality_pass(self, sample_ticks: list[OHLCVTick]) -> None:
        monitor = DataQualityMonitor()
        for tick in sample_ticks:
            report = monitor.check(tick)
            assert report.ticker == "INTG.NS"

    def test_feature_computation(self, sample_ticks: list[OHLCVTick]) -> None:
        buf      = TickerStateBuffer("INTG.NS", max_size=200)
        computer = RealTimeFeatureComputer()
        fv: FeatureVector | None = None
        for tick in sample_ticks:
            buf.push(tick)
            if buf.ready:
                fv = computer.compute(buf, tick)
        assert fv is not None
        assert fv.ticker == "INTG.NS"
        assert "close"   in fv.features
        assert "rsi_14"  in fv.features

    def test_prediction_ci_bounds_valid(self, sample_ticks: list[OHLCVTick]) -> None:
        # FIX: drift_score has default=0.0 in Prediction so it's not required
        pred = Prediction(
            ticker="INTG.NS",
            timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            predicted_close=2850.0,
            confidence_pct=78.5,
            prediction_lower=2790.0,
            prediction_upper=2910.0,
            model_version="3",
            model_name="gradient_boosting",
            feature_version="v2.1.0",
            schema_hash=_SCHEMA_HASH,
            # drift_score is optional (defaults to 0.0)
        )
        assert pred.prediction_lower < pred.predicted_close < pred.prediction_upper

    def test_prediction_invalid_ci_raises(self) -> None:
        """CI lower > point estimate must be rejected."""
        with pytest.raises(Exception):
            Prediction(
                ticker="INTG.NS",
                timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                predicted_close=2850.0,
                confidence_pct=78.5,
                prediction_lower=2900.0,   # <- lower > point: invalid
                prediction_upper=2910.0,
                model_version="3",
                model_name="gradient_boosting",
                feature_version="v2.1.0",
                schema_hash=_SCHEMA_HASH,
            )

    def test_prediction_with_drift_score(self) -> None:
        pred = Prediction(
            ticker="INTG.NS",
            timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            predicted_close=2850.0,
            confidence_pct=78.5,
            prediction_lower=2790.0,
            prediction_upper=2910.0,
            model_version="3",
            model_name="gradient_boosting",
            feature_version="v2.1.0",
            schema_hash=_SCHEMA_HASH,
            drift_score=0.12,
        )
        assert pred.drift_score == 0.12

    def test_quality_report_structure(self) -> None:
        report = DataQualityReport(ticker="INTG.NS")
        assert report.ticker  == "INTG.NS"
        assert report.passed  is True      # no checks = all pass
        assert report.failed_checks == []

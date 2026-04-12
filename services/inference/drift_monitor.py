"""Drift monitor — PSI-based feature drift detection."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    mn = min(float(np.min(expected)), float(np.min(actual)))
    mx = max(float(np.max(expected)), float(np.max(actual)))
    if mx == mn:
        return 0.0
    edges = np.linspace(mn, mx, buckets + 1)
    e_counts = np.histogram(expected, bins=edges)[0] + 1e-6
    a_counts = np.histogram(actual,   bins=edges)[0] + 1e-6
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))


def _psi_score(baseline: dict[str, list[float]], current: dict[str, float]) -> float:
    scores: list[float] = []
    for k, baseline_vals in baseline.items():
        if k not in current:
            continue
        expected = np.array(baseline_vals, dtype=float)
        actual   = np.array([current[k]], dtype=float)
        scores.append(_psi(expected, actual))
    return float(np.mean(scores)) if scores else 0.0


class DriftMonitor:
    def __init__(self) -> None:
        self._baseline: dict[str, list[float]] = {}

    def save_baseline(
        self, feature_values: dict[str, list[float]], path: str
    ) -> None:
        self._baseline = feature_values
        Path(path).write_text(json.dumps(feature_values))
        log.info("Drift baseline saved", extra={"path": path, "features": len(feature_values)})

    def load_baseline(self, path: str) -> None:
        self._baseline = json.loads(Path(path).read_text())

    def score(self, current_features: dict[str, float]) -> float:
        if not self._baseline:
            return 0.0
        return _psi_score(self._baseline, current_features)

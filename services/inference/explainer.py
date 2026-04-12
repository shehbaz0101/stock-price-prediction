"""SHAP explainer. Casts feature_values.get('ticker') to str explicitly."""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    feature_name: str
    feature_value: float
    shap_value: float
    abs_impact: float


@dataclass
class PredictionExplanation:
    ticker: str
    base_value: float
    predicted_value: float
    contributions: list[FeatureContribution] = field(default_factory=list)
    top_drivers: list[str] = field(default_factory=list)
    waterfall_b64: str = ""
    confidence_note: str = ""


class PredictionExplainer:
    def __init__(self, model: Any) -> None:
        self._model = model

    def explain(
        self,
        feature_values: dict[str, float | str],
        predicted_value: float,
        base_val: float,
    ) -> PredictionExplanation | None:
        try:
            import shap   # type: ignore[import]
            import numpy as np

            numeric = {
                k: float(v)
                for k, v in feature_values.items()
                if k != "ticker" and isinstance(v, (int, float))
            }
            X = np.array([list(numeric.values())])
            explainer    = shap.TreeExplainer(self._model)
            shap_values  = explainer.shap_values(X)
            shap_vec: list[float] = shap_values[0].tolist()

            contributions: list[FeatureContribution] = [
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(val),
                    shap_value=sv,
                    abs_impact=abs(sv),
                )
                for (name, val), sv in zip(numeric.items(), shap_vec)
            ]
            contributions.sort(key=lambda c: c.abs_impact, reverse=True)

            top_drivers    = self._build_drivers(contributions[:3], predicted_value, base_val)
            confidence_note = self._build_confidence(shap_vec, contributions)
            waterfall_b64   = self._build_waterfall(shap_vec, base_val, predicted_value)

            return PredictionExplanation(
                # FIX: feature_values.get("ticker", "") returns float | str
                # Cast to str to satisfy PredictionExplanation.ticker: str
                ticker=str(feature_values.get("ticker", "")),
                base_value=base_val,
                predicted_value=predicted_value,
                contributions=contributions,
                top_drivers=top_drivers,
                waterfall_b64=waterfall_b64,
                confidence_note=confidence_note,
            )
        except Exception as exc:
            log.error("SHAP explanation failed", extra={"exc": str(exc)})
            return None

    def _build_drivers(
        self, top3: list[FeatureContribution], pred: float, base: float
    ) -> list[str]:
        result: list[str] = []
        for c in top3:
            d = "pushed up" if c.shap_value > 0 else "pushed down"
            result.append(f"{c.feature_name}={c.feature_value:.4f} {d} by {abs(c.shap_value):.2f}")
        return result

    def _build_confidence(
        self, shap_vec: list[float], contributions: list[FeatureContribution]
    ) -> str:
        if not contributions:
            return "No contributions computed"
        total = sum(c.abs_impact for c in contributions) or 1.0
        dom   = contributions[0].abs_impact / total
        if dom > 0.6:
            return f"Dominated by {contributions[0].feature_name} ({dom:.0%})"
        return "Driven by a mix of features"

    def _build_waterfall(
        self, shap_vec: list[float], base_val: float, pred: float
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(shap_vec)), shap_vec)
            ax.set_title(f"SHAP waterfall  base={base_val:.1f} → pred={pred:.1f}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

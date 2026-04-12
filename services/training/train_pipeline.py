"""Training pipeline. All sklearn/numpy types fixed. mlflow submodules imported."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.models   # FIX: must import explicitly
import mlflow.sklearn  # FIX: must import explicitly
import mlflow.tracking # FIX: must import explicitly
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shared.contracts import FEATURE_KEYS, _SCHEMA_HASH
from services.inference.drift_monitor import DriftMonitor

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    mae: float
    rmse: float
    r2: float
    mape: float
    dir_acc: float
    n_train: int
    n_test: int


@dataclass
class TrainingConfig:
    model_name: str = "stock-predictor"
    output_dir: str = "/tmp/stock_model"
    feature_keys: list[str] = field(default_factory=lambda: list(FEATURE_KEYS))
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    random_state: int = 42


class StockModelTrainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg   = cfg
        self.drift = DriftMonitor()

    async def run(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray,  y_test: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_sync, X_train, y_train, X_test, y_test
        )

    def _run_sync(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> str:
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        champion_run_id:    str | None = None
        champion_cv_rmse:   float      = float("inf")
        champion_model_name: str       = ""

        with mlflow.start_run(run_name="stock_model_training") as run:
            run_id = run.info.run_id
            name   = "gradient_boosting"

            model = GradientBoostingRegressor(
                n_estimators  = self.cfg.n_estimators,
                learning_rate = self.cfg.learning_rate,
                max_depth     = self.cfg.max_depth,
                subsample     = self.cfg.subsample,
                random_state  = self.cfg.random_state,
            )

            # FIX: cast to np.ndarray so sklearn type stubs are satisfied
            Xtr = np.asarray(X_train)
            ytr = np.asarray(y_train)
            Xte = np.asarray(X_test)
            yte = np.asarray(y_test)

            model.fit(Xtr, ytr)                          # type: ignore[arg-type]
            preds: np.ndarray = model.predict(Xte)       # type: ignore[arg-type]

            # Metrics — FIX: asarray + cast to float
            mae  = float(mean_absolute_error(yte, preds))            # type: ignore[arg-type]
            rmse = float(np.sqrt(mean_squared_error(yte, preds)))    # type: ignore[arg-type]
            r2   = float(r2_score(yte, preds))                       # type: ignore[arg-type]
            mape = float(
                np.mean(np.abs(yte - preds) / np.maximum(np.abs(yte), 1e-6)) * 100
            )

            # Directional accuracy
            if len(yte) > 1:
                actual_dir = np.sign(np.diff(yte))    # type: ignore[arg-type]
                pred_dir   = np.sign(np.diff(preds))  # type: ignore[arg-type]
                dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)
            else:
                dir_acc = 50.0

            mlflow.log_metrics({
                "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "dir_acc": dir_acc,
            })
            mlflow.set_tag("feature_schema_hash", _SCHEMA_HASH)

            # Feature importance plot
            fi_path = self._plot_fi(model, name)
            if fi_path:
                mlflow.log_artifact(fi_path, artifact_path="plots")

            # Log model — FIX: mlflow.sklearn + mlflow.models imported at top
            mlflow.sklearn.log_model(          # type: ignore[attr-defined]
                model,
                artifact_path="model",
                registered_model_name=self.cfg.model_name,
                signature=mlflow.models.infer_signature(Xtr, ytr),  # type: ignore[attr-defined]
                input_example=Xtr[:5],
            )

            if rmse < champion_cv_rmse:
                champion_cv_rmse    = rmse
                champion_run_id     = run_id
                champion_model_name = name

        if champion_run_id:
            self._promote_champion(champion_run_id, champion_model_name, champion_cv_rmse)

        # Save drift baseline
        baseline_path = str(Path(self.cfg.output_dir) / "training_baseline.json")
        training_feature_values: dict[str, list[float]] = {
            k: Xtr[:, i].tolist()
            for i, k in enumerate(self.cfg.feature_keys)
            if i < Xtr.shape[1]
        }
        self.drift.save_baseline(training_feature_values, baseline_path)
        log.info("Drift baseline saved", extra={"path": baseline_path})

        return champion_run_id or ""

    def _plot_fi(self, model: GradientBoostingRegressor, name: str) -> str | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # FIX: feature_importances_ can be typed as _Array1D|float64|NDArray —
            # cast to plain ndarray so len() is unambiguously valid
            importances: np.ndarray = np.asarray(model.feature_importances_)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(self.cfg.feature_keys[:int(importances.size)],
                    importances, color="steelblue")
            ax.set_title(f"{name} feature importance")
            path = f"{self.cfg.output_dir}/fi_{name}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception:
            return None

    def _promote_champion(
        self, run_id: str, model_name: str, cv_rmse: float
    ) -> None:
        # FIX: mlflow.tracking imported explicitly above
        client   = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if not versions:
            log.warning("No model version for champion run", extra={"run_id": run_id})
            return
        version = versions[0].version
        client.transition_model_version_stage(
            name=model_name, version=version,
            stage="Production", archive_existing_versions=True,
        )
        log.info("Champion promoted", extra={
            "model": model_name, "version": version, "cv_rmse": cv_rmse,
        })

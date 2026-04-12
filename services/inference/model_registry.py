"""MLflow model registry. mlflow.sklearn and mlflow.models imported explicitly."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlflow
import mlflow.models   # FIX: submodule must be imported explicitly
import mlflow.sklearn  # FIX: submodule must be imported explicitly
import numpy as np
from mlflow.tracking import MlflowClient

from shared.contracts import _SCHEMA_HASH

log = logging.getLogger(__name__)


class MLflowModelRegistry:
    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        stage: str = "Production",
    ) -> None:
        self.tracking_uri  = tracking_uri
        self.model_name    = model_name
        self.stage         = stage
        self.current_version: str = ""
        self.feature_schema_hash: str = _SCHEMA_HASH
        self._model: Any | None = None
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def sync_load(self) -> None:
        """Load model synchronously — call once from lifespan."""
        self._do_load()

    def _do_load(self) -> None:
        versions = self._client.get_latest_versions(self.model_name, stages=[self.stage])
        if not versions:
            log.warning("No Production model found — using fallback")
            return

        v = versions[0]
        model_uri = f"models:/{self.model_name}/{self.stage}"
        log.info("Loading model", extra={"uri": model_uri})

        # FIX: mlflow.sklearn must be imported above
        self._model = mlflow.sklearn.load_model(model_uri)  # type: ignore[attr-defined]
        self.current_version = v.version

        if v.run_id:
            try:
                tags = self._client.get_run(v.run_id).data.tags
                self.feature_schema_hash = tags.get("feature_schema_hash", _SCHEMA_HASH)
            except Exception as exc:
                log.warning("Could not read schema hash tag", extra={"exc": str(exc)})

        log.info("Model loaded", extra={
            "version": self.current_version,
            "feature_hash": self.feature_schema_hash,
        })

    async def reload_if_new(self) -> bool:
        """Check MLflow for a newer version; reload if found."""
        versions = self._client.get_latest_versions(self.model_name, stages=[self.stage])
        if not versions:
            return False
        latest = versions[0].version
        if latest != self.current_version:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._do_load)
            return True
        return False

    def predict(self, features: dict[str, float]) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        X = np.array([[features.get(k, 0.0) for k in sorted(features)]])
        return float(self._model.predict(X)[0])  # type: ignore[index]

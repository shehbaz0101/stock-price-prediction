"""S3-backed offline feature store. Runs boto3 uploads in executor."""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class OfflineFeatureStore:
    def __init__(self, s3_bucket: str, s3_prefix: str = "features") -> None:
        self._bucket = s3_bucket
        self._prefix = s3_prefix

    async def flush_to_offline(
        self, ticker: str, vectors: list[Any], date: str
    ) -> str:
        if not vectors:
            log.warning("No vectors to flush", extra={"ticker": ticker})
            return ""

        rows: list[dict[str, Any]] = []
        for v in vectors:
            row: dict[str, Any] = {
                "ticker": v.ticker,
                "timestamp_utc": v.timestamp_utc.isoformat(),
            }
            # FIX: v.features may be a Pydantic model — use model_dump()
            features_dict: dict[str, Any] = (
                v.features.model_dump() if hasattr(v.features, "model_dump")
                else dict(v.features)
            )
            row.update(features_dict)
            rows.append(row)

        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            local_path = f.name

        df.to_parquet(local_path, index=False, engine="pyarrow", compression="snappy")
        s3_key = f"{self._prefix}/{ticker}/{date}.parquet"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._upload_sync, local_path, s3_key)
        Path(local_path).unlink(missing_ok=True)

        log.info("Flushed to offline store",
                 extra={"ticker": ticker, "s3_key": s3_key, "rows": len(df)})
        return s3_key

    def _upload_sync(self, local_path: str, s3_key: str) -> None:
        import boto3
        boto3.client("s3").upload_file(local_path, self._bucket, s3_key)

    async def read_offline(self, ticker: str, date: str) -> pd.DataFrame:
        s3_key = f"{self._prefix}/{ticker}/{date}.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            local_path = f.name
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._download_sync, s3_key, local_path)
        df = pd.read_parquet(local_path)
        Path(local_path).unlink(missing_ok=True)
        return df

    def _download_sync(self, s3_key: str, local_path: str) -> None:
        import boto3
        boto3.client("s3").download_file(self._bucket, s3_key, local_path)

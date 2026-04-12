"""Async Kafka producer wrapper."""
from __future__ import annotations

import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

log = logging.getLogger(__name__)


class KafkaProducerClient:
    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    def _get(self) -> AIOKafkaProducer:
        if self._producer is None:
            raise RuntimeError("Producer not started — call await start() first")
        return self._producer

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode() if k else None,
        )
        await self._get().start()
        log.info("Kafka producer started")

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, payload: Any, key: str | None = None) -> None:
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        await self._get().send_and_wait(topic, value=data, key=key)

    async def publish_to_dlq(self, topic: str, payload: Any) -> None:
        await self.publish(topic, payload)

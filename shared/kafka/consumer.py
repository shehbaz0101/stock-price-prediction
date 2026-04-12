"""Async Kafka consumer wrapper. Works with Python 3.12 (asyncio.coroutine removed)."""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

from aiokafka import AIOKafkaConsumer

log = logging.getLogger(__name__)
T = TypeVar("T")


class KafkaConsumerClient:
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        auto_offset_reset: str = "latest",
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset

    async def stream(
        self,
        topic: str,
        schema: Any,
        on_error: Callable[..., Any] | None = None,
    ) -> AsyncGenerator[Any, None]:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            auto_offset_reset=self._auto_offset_reset,
            enable_auto_commit=False,
        )
        await consumer.start()
        log.info("Kafka consumer started", extra={"topic": topic, "group": self._group_id})
        try:
            async for msg in consumer:
                try:
                    event = schema.model_validate(msg.value)
                    await consumer.commit()
                    yield event
                except Exception as exc:
                    log.error(
                        "Message deserialization failed",
                        extra={"exc": str(exc), "topic": topic, "raw": str(msg.value)[:200]},
                    )
                    if on_error is not None:
                        # asyncio.coroutine() was removed in Python 3.11.
                        # Check iscoroutinefunction and await directly.
                        if inspect.iscoroutinefunction(on_error):
                            await on_error(exc, msg.value)
                        else:
                            on_error(exc, msg.value)
        finally:
            await consumer.stop()

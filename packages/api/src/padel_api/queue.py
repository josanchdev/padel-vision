"""Job enqueuing.

A `JobQueue` hands a match id to a worker. The Arq implementation is used in
production; the in-memory one records enqueued ids so tests can assert without
a running worker or Redis.
"""

from __future__ import annotations

from typing import Protocol

from arq import ArqRedis

PROCESS_TASK = "process_match"


class JobQueue(Protocol):
    async def enqueue(self, match_id: str) -> None: ...


class InMemoryJobQueue:
    """Records enqueued match ids without running anything (tests)."""

    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue(self, match_id: str) -> None:
        self.enqueued.append(match_id)


class ArqJobQueue:
    def __init__(self, redis: ArqRedis) -> None:
        self._redis = redis

    async def enqueue(self, match_id: str) -> None:
        await self._redis.enqueue_job(PROCESS_TASK, match_id)

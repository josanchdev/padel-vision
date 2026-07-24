"""Match metadata storage.

A `MatchStore` persists `Match` records so the API and worker share state. The
Redis implementation is used in production; the in-memory one keeps tests free
of infrastructure. Both satisfy the same Protocol.
"""

from __future__ import annotations

from typing import Protocol

import redis.asyncio as redis

from padel_api.models import Match

_KEY_PREFIX = "match:"
_INDEX_KEY = "matches"


class MatchStore(Protocol):
    async def save(self, match: Match) -> None: ...

    async def get(self, match_id: str) -> Match | None: ...

    async def list(self) -> list[Match]: ...


class InMemoryMatchStore:
    """Non-persistent store for tests and local runs without Redis."""

    def __init__(self) -> None:
        self._matches: dict[str, str] = {}

    async def save(self, match: Match) -> None:
        match.touch()
        self._matches[match.id] = match.model_dump_json()

    async def get(self, match_id: str) -> Match | None:
        raw = self._matches.get(match_id)
        return Match.model_validate_json(raw) if raw else None

    async def list(self) -> list[Match]:
        matches = [Match.model_validate_json(raw) for raw in self._matches.values()]
        return sorted(matches, key=lambda m: m.created_at, reverse=True)


class RedisMatchStore:
    """Redis-backed store; matches live as JSON strings under match:<id>."""

    def __init__(self, client: redis.Redis) -> None:
        self._redis = client

    async def save(self, match: Match) -> None:
        match.touch()
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(_KEY_PREFIX + match.id, match.model_dump_json())
            pipe.zadd(_INDEX_KEY, {match.id: match.created_at.timestamp()})
            await pipe.execute()

    async def get(self, match_id: str) -> Match | None:
        raw = await self._redis.get(_KEY_PREFIX + match_id)
        return Match.model_validate_json(raw) if raw else None

    async def list(self) -> list[Match]:
        ids = await self._redis.zrevrange(_INDEX_KEY, 0, -1)
        if not ids:
            return []
        raws = await self._redis.mget(_KEY_PREFIX + mid for mid in ids)
        return [Match.model_validate_json(raw) for raw in raws if raw]

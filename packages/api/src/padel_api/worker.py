"""Arq worker: runs the CV pipeline for a queued match, updating its status.

The pipeline is synchronous (OpenCV + Torch), so it runs in a thread. Progress
is written back from that thread with a synchronous Redis client, while the
async task owns the pending->processing->done/failed transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast

import redis
import redis.asyncio as aioredis
from arq.connections import RedisSettings

from padel_api.config import get_settings
from padel_api.models import Match, MatchStatus
from padel_api.store import RedisMatchStore

_KEY_PREFIX = "match:"


def _write_progress(sync_redis: redis.Redis, match_id: str, progress: float) -> None:
    """Read-modify-write only the progress field (throttled, mid-run only)."""
    # redis-py types get() as a sync/async union (ResponseT); this is a sync client.
    raw = cast("str | None", sync_redis.get(_KEY_PREFIX + match_id))
    if raw is None:
        return
    match = Match.model_validate_json(raw)
    match.progress = progress
    match.touch()
    sync_redis.set(_KEY_PREFIX + match_id, match.model_dump_json())


def _run_pipeline(match_id: str, redis_url: str) -> tuple[int, float | None]:
    """Synchronous body in a worker thread. Returns (shots detected, duration s)."""
    from padel_cv.cli import process_video

    settings = get_settings()
    sync_redis = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
    try:
        court_model = str(settings.court_model) if settings.court_model else None
        result_path = settings.results_dir / f"{match_id}.mp4"
        result = process_video(
            input_path=settings.uploads_dir / f"{match_id}.mp4",
            output_path=result_path,
            model_name="yolo26n-pose.pt",
            confidence=0.3,
            image_size=1920,
            max_frames=None,
            court_model=court_model,
            on_progress=lambda p: _write_progress(sync_redis, match_id, p),
            data_out=settings.data_dir / match_id,  # writes {id}.json and {id}.csv (ADR-0010)
        )
        # Dashboard card assets: a thumbnail from the processed video + duration.
        from padel_cv.thumbnail import write_thumbnail

        write_thumbnail(result_path, settings.data_dir / f"{match_id}.jpg")
        duration_s = _read_duration(settings.data_dir / f"{match_id}.json", result.frames_written)
        return result.shots_detected, duration_s
    finally:
        sync_redis.close()


def _read_duration(data_json: Path, frames: int) -> float | None:
    """Video length in seconds = frames / fps (fps from the data JSON)."""
    import json

    if not data_json.exists() or frames <= 0:
        return None
    fps = json.loads(data_json.read_text()).get("fps") or 30.0
    return frames / float(fps)


async def process_match(ctx: dict[str, Any], match_id: str) -> None:
    import asyncio

    store: RedisMatchStore = ctx["store"]
    redis_url = ctx["redis_url"]
    match = await store.get(match_id)
    if match is None:
        return
    match.status = MatchStatus.PROCESSING
    await store.save(match)
    try:
        shots, duration_s = await asyncio.to_thread(_run_pipeline, match_id, redis_url)
        match = await store.get(match_id) or match
        match.status = MatchStatus.DONE
        match.progress = 1.0
        match.shots_detected = shots
        match.duration_s = duration_s
        await store.save(match)
    except Exception as exc:
        match = await store.get(match_id) or match
        match.status = MatchStatus.FAILED
        match.error = str(exc)
        await store.save(match)


async def _on_startup(ctx: dict[str, Any]) -> None:
    redis_url = str(get_settings().redis_url)
    ctx["redis_url"] = redis_url
    ctx["async_redis"] = aioredis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
    ctx["store"] = RedisMatchStore(ctx["async_redis"])


async def _on_shutdown(ctx: dict[str, Any]) -> None:
    await ctx["async_redis"].aclose()


class WorkerSettings:
    """Arq entrypoint: `arq padel_api.worker.WorkerSettings`."""

    functions: ClassVar = [process_match]
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    redis_settings = RedisSettings.from_dsn(str(get_settings().redis_url))
    max_jobs = 1  # single GPU: process one match at a time
    job_timeout = 3600

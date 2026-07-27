"""Padel Vision API: submit a match video, track its analysis job, get results."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from padel_api.config import Settings, get_settings
from padel_api.models import Match, MatchStatus
from padel_api.queue import ArqJobQueue, JobQueue
from padel_api.store import MatchStore, RedisMatchStore

UPLOAD_CHUNK = 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    redis_url = str(settings.redis_url)
    app.state.redis = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
    app.state.store = RedisMatchStore(app.state.redis)
    app.state.arq = await create_pool(RedisSettings.from_dsn(redis_url))
    app.state.queue = ArqJobQueue(app.state.arq)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.arq.aclose()


app = FastAPI(title="Padel Vision API", version="0.1.0", lifespan=lifespan)

# Serve the compiled web bundle's assets (Vite output; ADR-0011). Mounted only
# when present so tests and API-only runs don't require a build.
_ASSETS_DIR = STATIC_DIR / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


def get_store() -> MatchStore:
    store: MatchStore = app.state.store
    return store


def get_queue() -> JobQueue:
    queue: JobQueue = app.state.queue
    return queue


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/matches", response_model=Match, status_code=201)
async def submit_match(
    video: UploadFile,
    settings: Settings = Depends(get_settings),
    store: MatchStore = Depends(get_store),
    queue: JobQueue = Depends(get_queue),
) -> Match:
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="Upload must be a video file")
    match_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{match_id}.mp4"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await video.read(UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                await out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds size limit")
            await out.write(chunk)
    match = Match(id=match_id, filename=video.filename or f"{match_id}.mp4")
    await store.save(match)
    await queue.enqueue(match_id)
    return match


@app.get("/matches", response_model=list[Match])
async def list_matches(store: MatchStore = Depends(get_store)) -> list[Match]:
    return await store.list()


@app.get("/matches/{match_id}", response_model=Match)
async def get_match(match_id: str, store: MatchStore = Depends(get_store)) -> Match:
    match = await store.get(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@app.get("/matches/{match_id}/result")
async def get_result(
    match_id: str,
    settings: Settings = Depends(get_settings),
    store: MatchStore = Depends(get_store),
) -> FileResponse:
    match = await store.get(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status is not MatchStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Match is {match.status.value}")
    result = settings.results_dir / f"{match_id}.mp4"
    if not result.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    # No filename= so the browser plays it inline in a <video> tag.
    return FileResponse(result, media_type="video/mp4")


@app.get("/matches/{match_id}/thumbnail")
async def get_thumbnail(
    match_id: str,
    settings: Settings = Depends(get_settings),
    store: MatchStore = Depends(get_store),
) -> FileResponse:
    """A still from the processed video for the dashboard match card."""
    match = await store.get(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    thumb = settings.data_dir / f"{match_id}.jpg"
    if not thumb.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/matches/{match_id}/data")
async def get_data(
    match_id: str,
    settings: Settings = Depends(get_settings),
    store: MatchStore = Depends(get_store),
) -> FileResponse:
    """Structured analysis JSON — the queryable product the web consumes (ADR-0010)."""
    match = await store.get(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status is not MatchStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Match is {match.status.value}")
    data = settings.data_dir / f"{match_id}.json"
    if not data.exists():
        raise HTTPException(status_code=404, detail="Data file missing")
    return FileResponse(data, media_type="application/json")


@app.get("/matches/{match_id}/clip")
async def get_clip(
    match_id: str,
    frame: int,
    settings: Settings = Depends(get_settings),
    store: MatchStore = Depends(get_store),
) -> FileResponse:
    """A short clip (3s each side) around a frame, cut on-demand from the result.

    Powers the shot-table modal: click a shot -> see it in context, skeleton and
    label baked in. Cut lazily (not pre-generated) to avoid a file per shot.
    """
    import asyncio

    from padel_cv.clip_export import cut_clip

    match = await store.get(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status is not MatchStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Match is {match.status.value}")
    result = settings.results_dir / f"{match_id}.mp4"
    if not result.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    fps = _read_fps(settings.data_dir / f"{match_id}.json")
    clip_path = settings.data_dir / f"{match_id}_clip_{frame}.mp4"
    if not clip_path.exists():
        try:
            await asyncio.to_thread(cut_clip, result, clip_path, frame, fps)
        except Exception as exc:  # ffmpeg failure
            raise HTTPException(status_code=500, detail=f"Clip generation failed: {exc}") from exc
    return FileResponse(clip_path, media_type="video/mp4")


def _read_fps(data_json: Path, default: float = 30.0) -> float:
    """Read the match fps from its data JSON, falling back to a sane default."""
    import json

    if not data_json.exists():
        return default
    fps = json.loads(data_json.read_text()).get("fps")
    return float(fps) if fps else default

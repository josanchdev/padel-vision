"""Export analysed rallies as the static files the web viewer reads (ADR-0017).

The viewer has no backend: it reads plain files. Per point, under
`<points_dir>/<id>/`:

    point.json    shots and player positions (contract below)
    video.mp4     the annotated rally, 1080p H.264 with the original sound
    thumb.jpg     a frame from the middle of the rally
    preview.mp4   a few seconds of the annotated video, small, for hover

and one `<points_dir>/index.json` listing every point for the home grid.

point.json, schema_version 2:

    id, title, city, date (ISO or null), fps, frames, duration_s
    shots    [{frame, t, player (1-4 or null), type (Forehand, Backhand, Smash,
             Serve or null), confidence}]
    players  {"1": [[frame, x_m, y_m], ...], ...} in court metres: x across the
             court, y from the camera's baseline (0) to the far one (20). The
             viewer draws y flipped so the camera side sits at the bottom, as
             it does in the video.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg

from padel_ml.rally_analysis import RallyAnalysis
from padel_ml.rally_render import render_rally

SCHEMA_VERSION = 2
PREVIEW_SECONDS = 6.0
PREVIEW_WIDTH = 640

_CVSPORTS_ID = re.compile(r"^(?P<date>\d{8})_(?P<place>[A-Z]+)_(?P<index>\d+)$")
CITIES = {
    "VIGO": "Vigo",
    "BRUSSEL": "Bruselas",
    "VALLADOLID": "Valladolid",
    "VALENCIA": "Valencia",
    "MALAGA": "Málaga",
    "FINLAND": "Finlandia",
    "MADRID": "Madrid",
    "DUITSLAND": "Alemania",
    "AMSTERDAM": "Ámsterdam",
    "MENORCA": "Menorca",
    "MALMO": "Malmö",
}
"""CVSPORTS names its tournaments in several languages; the viewer speaks Spanish."""


@dataclass(frozen=True)
class PointMeta:
    title: str
    city: str | None
    date: str | None
    """ISO date, or None when the video name does not carry one."""


def describe(stem: str, title: str | None = None) -> PointMeta:
    """A readable name: "20230528_VIGO_01" -> "Punto 01 · Vigo", 2023-05-28.

    Any other video keeps its file name unless a title is given.
    """
    match = _CVSPORTS_ID.match(stem)
    if match is None:
        return PointMeta(title or stem, None, None)
    raw = match.group("date")
    city = CITIES.get(match.group("place"), match.group("place").title())
    iso = date(int(raw[:4]), int(raw[4:6]), int(raw[6:])).isoformat()
    return PointMeta(title or f"Punto {match.group('index')} · {city}", city, iso)


def point_record(point_id: str, meta: PointMeta, analysis: RallyAnalysis) -> dict[str, Any]:
    """The point.json contract, from an analysis."""
    players: dict[str, list[list[float]]] = {}
    for frame in sorted(analysis.players):
        for pose in analysis.players[frame]:
            if pose.player_id is None or pose.court_position_m is None:
                continue
            x, y = pose.court_position_m
            players.setdefault(str(pose.player_id), []).append([frame, round(x, 2), round(y, 2)])
    return {
        "schema_version": SCHEMA_VERSION,
        "id": point_id,
        "title": meta.title,
        "city": meta.city,
        "date": meta.date,
        "fps": analysis.fps,
        "frames": analysis.n_frames,
        "duration_s": round(analysis.n_frames / analysis.fps, 2),
        "shots": [
            {
                "frame": shot.frame_index,
                "t": round(shot.timestamp_s, 3),
                "player": shot.player_id,
                "type": shot.shot_type,
                "confidence": round(shot.confidence, 3),
            }
            for shot in analysis.shots
        ],
        "players": players,
    }


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *args], check=True
    )


def export_point(analysis: RallyAnalysis, points_dir: Path, meta: PointMeta) -> Path:
    """Write one point's folder; returns it."""
    point_id = analysis.video.stem
    folder = points_dir / point_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "point.json").write_text(
        json.dumps(
            point_record(point_id, meta, analysis), ensure_ascii=False, separators=(",", ":")
        )
    )

    # The overlay is drawn without sound; put the original track back, since
    # the racket's pop is half of what makes a rally readable. Faststart so the
    # browser can start playing before the whole file has arrived.
    silent = folder / "video.silent.mp4"
    render_rally(analysis, silent)
    _ffmpeg(
        "-i", str(silent), "-i", str(analysis.video),
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart", str(folder / "video.mp4"),
    )  # fmt: skip
    silent.unlink()

    middle = _middle_of_play(analysis)
    capture = cv2.VideoCapture(str(analysis.video))
    capture.set(cv2.CAP_PROP_POS_MSEC, middle * 1000)
    ok, frame = capture.read()
    capture.release()
    if ok:
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (960, round(960 * height / width)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / "thumb.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 86])

    start = max(middle - PREVIEW_SECONDS / 2, 0.0)
    _ffmpeg(
        "-ss", f"{start:.2f}", "-t", f"{PREVIEW_SECONDS}", "-i", str(folder / "video.mp4"),
        "-an", "-vf", f"scale={PREVIEW_WIDTH}:-2", "-c:v", "libx264", "-crf", "28",
        "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(folder / "preview.mp4"),
    )  # fmt: skip
    return folder


def _middle_of_play(analysis: RallyAnalysis) -> float:
    """The instant of the middle shot: the thumbnail shows play, not the walk to serve."""
    if analysis.shots:
        return analysis.shots[len(analysis.shots) // 2].timestamp_s
    return analysis.n_frames / analysis.fps / 2


def write_index(points_dir: Path) -> Path:
    """index.json: one summary per exported point, in date and then name order."""
    summaries = []
    for path in sorted(points_dir.glob("*/point.json")):
        point = json.loads(path.read_text())
        summaries.append(
            {
                "id": point["id"],
                "title": point["title"],
                "city": point["city"],
                "date": point["date"],
                "duration_s": point["duration_s"],
                "shots": len(point["shots"]),
            }
        )
    summaries.sort(key=lambda s: (s["date"] or "", s["id"]))
    out = points_dir / "index.json"
    out.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "points": summaries}, ensure_ascii=False, indent=1
        )
    )
    return out

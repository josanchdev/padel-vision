"""Structured match analysis: the canonical data product (ADR-0010).

The pipeline generates rich per-frame information (player positions in metres,
shots, ball, bounces) that a coach can actually act on — far more useful than an
annotated video, which is only visual verification. This module turns that
stream into a structured, queryable record: the real product.

`MatchAnalysis` is the canonical container. It serializes to JSON (rich, nested,
what the API/web consume) and exports to CSV (one row per event, what a coach
opens in Excel). A `schema_version` makes the JSON a versioned contract so the
web knows what to expect and the format can evolve without breaking clients.

Populated by accumulating from processed frames (see `accumulate_frame`); the
bounces come from the trajectory analysis at the end of the run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from padel_cv.bounces import Bounce
from padel_cv.pipeline import Frame

SCHEMA_VERSION = 1


@dataclass
class PlayerFrameRecord:
    """One player's state in one frame."""

    frame_index: int
    timestamp_s: float
    player_id: int
    track_id: int | None
    court_x_m: float | None
    court_y_m: float | None
    on_court: bool | None


@dataclass
class ShotRecord:
    frame_index: int
    timestamp_s: float
    player_id: int
    label: str
    confidence: float


@dataclass
class BallRecord:
    frame_index: int
    timestamp_s: float
    image_x: float
    image_y: float
    court_x_m: float | None
    court_y_m: float | None
    confidence: float


@dataclass
class BounceRecord:
    frame_index: int
    image_x: float
    image_y: float


@dataclass
class MatchAnalysis:
    """All structured data extracted from one match video."""

    source_video: str
    fps: float
    schema_version: int = SCHEMA_VERSION
    players: list[PlayerFrameRecord] = field(default_factory=list)
    shots: list[ShotRecord] = field(default_factory=list)
    ball: list[BallRecord] = field(default_factory=list)
    bounces: list[BounceRecord] = field(default_factory=list)

    def accumulate_frame(self, frame: Frame) -> None:
        """Append this frame's detections to the record."""
        for pose in frame.poses:
            if pose.player_id is None:
                continue  # only track identified on-court players
            cx, cy = pose.court_position_m if pose.court_position_m is not None else (None, None)
            self.players.append(
                PlayerFrameRecord(
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    player_id=pose.player_id,
                    track_id=pose.track_id,
                    court_x_m=cx,
                    court_y_m=cy,
                    on_court=pose.on_court,
                )
            )
        for shot in frame.shot_events:
            self.shots.append(
                ShotRecord(
                    frame_index=shot.frame_index,
                    timestamp_s=frame.timestamp_s,
                    player_id=shot.player_id,
                    label=shot.label,
                    confidence=shot.confidence,
                )
            )
        if frame.ball is not None:
            bcx, bcy = frame.ball.court_xy_m if frame.ball.court_xy_m is not None else (None, None)
            self.ball.append(
                BallRecord(
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    image_x=frame.ball.image_xy[0],
                    image_y=frame.ball.image_xy[1],
                    court_x_m=bcx,
                    court_y_m=bcy,
                    confidence=frame.ball.confidence,
                )
            )

    def set_bounces(self, bounces: list[Bounce]) -> None:
        self.bounces = [BounceRecord(b.frame_index, b.x_px, b.y_px) for b in bounces]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "fps": self.fps,
            "players": [asdict(p) for p in self.players],
            "shots": [asdict(s) for s in self.shots],
            "ball": [asdict(b) for b in self.ball],
            "bounces": [asdict(b) for b in self.bounces],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def write_csv(self, path: Path) -> None:
        """Flatten every event to one row: event_type + shared columns.

        A long/tidy table (one row per event) is what opens cleanly in Excel and
        pivots easily — the coach's working format.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "event_type",
            "frame_index",
            "timestamp_s",
            "player_id",
            "label",
            "confidence",
            "court_x_m",
            "court_y_m",
            "image_x",
            "image_y",
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for p in self.players:
                writer.writerow({"event_type": "player", **asdict(p)})
            for s in self.shots:
                writer.writerow({"event_type": "shot", **asdict(s)})
            for b in self.ball:
                writer.writerow({"event_type": "ball", **asdict(b)})
            for bo in self.bounces:
                writer.writerow({"event_type": "bounce", **asdict(bo)})

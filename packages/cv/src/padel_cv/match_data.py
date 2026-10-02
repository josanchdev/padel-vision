"""Structured rally analysis: the data product behind the video (ADR-0010).

The analysis produces per-frame information (player positions in metres, shots,
ball, bounces) that is worth more than the annotated video, which is only visual
verification. `MatchAnalysis` holds it and serializes it to JSON; the
`schema_version` makes that JSON a versioned contract. `RallyAnalysis`
(`padel_ml.rally_analysis`) fills it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

"""Cached per-rally detections, read back as the pipeline produces them.

`scripts/extract_rally_features.py` runs pose + identity + ball once over every
CVSPORTS rally and stores the result, so experiments over the 99 rallies take
minutes instead of hours. This turns a cache file back into the same structures
`analyze_rally` builds: identified players per frame and the post-processed ball.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from padel_cv.cvsports import tournament_of
from padel_cv.detections import PoseDetection


@dataclass
class RallyFeatures:
    rally: str
    fps: float
    n_frames: int
    players: dict[int, list[PoseDetection]]
    """Identified players only, with the detector's own boxes."""
    ball: dict[int, tuple[float, float]]

    @property
    def tournament(self) -> str:
        """ "20230528_VIGO_03" -> "20230528_VIGO"."""
        return tournament_of(self.rally)


def load_rally_features(path: Path) -> RallyFeatures:
    data = np.load(path, allow_pickle=True)
    keypoints = data["keypoints"].item()
    player_ids = data["player_ids"].item()
    boxes = data["boxes"].item()
    players: dict[int, list[PoseDetection]] = {}
    for frame, frame_keypoints in keypoints.items():
        players[int(frame)] = [
            PoseDetection(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=1.0,
                keypoints=np.asarray(kp, dtype=np.float32),
                player_id=int(player_id),
            )
            for kp, box, player_id in zip(
                frame_keypoints, boxes.get(frame, []), player_ids.get(frame, []), strict=False
            )
            if player_id is not None and player_id > 0
        ]
    ball = {int(f): (float(x), float(y)) for f, (x, y) in data["ball"].item().items()}
    return RallyFeatures(path.stem, float(data["fps"]), int(data["n_frames"]), players, ball)

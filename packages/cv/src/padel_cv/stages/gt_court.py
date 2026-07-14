"""Court stage backed by PadelTracker100 ground-truth homographies.

DEV/EVAL ONLY: this stage reads precomputed per-frame homography matrices from
the PadelTracker100 annotations, so it only works on those two match videos.
It exists to unblock and validate everything downstream (court filtering,
minimap, metric positions) while the learned court-keypoint detector
(ADR-0001) is being built. The production replacement must compute H from
detected court keypoints on any video (ADR-0005).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from padel_cv.court import HomographyArray, localize_players
from padel_cv.pipeline import Frame


class GroundTruthCourtStage:
    """Attaches GT homography to frames and localizes players in meters."""

    def __init__(self, homography_json: str | Path) -> None:
        with open(homography_json) as f:
            entries = json.load(f)["homography"]
        # The file contains duplicate entries per image_id; last one wins.
        # image_id is 1-based and matches frame_index + 1.
        self._by_image_id: dict[int, HomographyArray] = {
            entry["image_id"]: np.array(entry["H"], dtype=np.float64) for entry in entries
        }

    def process(self, frame: Frame) -> Frame:
        homography = self._by_image_id.get(frame.index + 1)
        if homography is None:
            return frame
        frame.homography = homography
        localize_players(frame)
        return frame

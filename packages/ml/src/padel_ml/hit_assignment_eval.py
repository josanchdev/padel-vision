"""Per-frame state for measuring hit assignment against the paper (ADR-0015).

`build_states` runs pose + identity + ball over a rally — the same steps as
`analyze_rally` — and keeps, per frame, the identified players and the ball. The
evidence scripts score the vote on it at the ANNOTATED hit times, so the
assignment step is measured alone (`scripts/evidence_hit_assignment.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np

from padel_cv.pipeline import BallDetection, Frame, ImageArray, PoseDetection
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.hit_assignment import FrameState


def build_states(
    video: Path,
    pose_stage: PlayerPoseStage,
    ball_detector: BallDetector,
    court_corners_px: np.ndarray | None,
    fps: float,
) -> dict[int, FrameState]:
    """Run pose + ball + identity over a rally and collect per-frame state.

    Starts from clean detectors, as `analyze_rally` does: the pose tracker and
    the ball buffer would otherwise carry the previous rally into this one.
    """
    pose_stage.reset()
    ball_detector.reset()
    polygon = court_mask_polygon(court_corners_px) if court_corners_px is not None else None
    identity = PlayerIdentityTracker(fps=fps)
    capture = cv2.VideoCapture(str(video))
    poses_by_frame: dict[int, list[PoseDetection]] = {}
    raw_ball: list[BallHit] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = pose_stage.process(
            Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
        )
        players = filter_players(frame.poses, polygon)
        identity.update(index, players)
        # keep every pose object: the identity tracker back-fills IDs on the
        # startup window once numbering is fixed, so filtering here would drop
        # poses that are about to be identified.
        poses_by_frame[index] = players
        hit = ball_detector.detect(image, index)
        if hit is not None:
            raw_ball.append(hit)
        index += 1
    capture.release()

    ball_by_frame = {
        h.frame_index: BallDetection((h.x_px, h.y_px), h.confidence)
        for h in postprocess_ball(raw_ball)
    }
    return {
        i: FrameState(
            poses=[p for p in poses_by_frame.get(i, []) if p.player_id is not None],
            ball=ball_by_frame.get(i),
        )
        for i in range(index)
    }

"""Assign a detected hit to the player who made it (ADR-0015, replica of Decorte 2024).

The audio detector says WHEN a hit happens; this says WHO. Replicates the paper's
method (accuracy 83.7% player / 86.8% team), not a naive 1-frame nearest-wrist:

- take a window of ~500 ms (12 frames @25fps) around the hit,
- per frame with a ball detection, measure ball→player distance (min of both
  wrists if pose present, else bbox centre),
- weighted majority vote across the window, weight by standardized distance (eq.1),
- tie-break by smallest euclidean distance over all frames.

Multi-frame voting is robust to the frames where pose or ball is momentarily
missing — the single-frame version we tried before was fragile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from padel_cv.pipeline import BallDetection, PoseDetection

_L_WRIST, _R_WRIST = 9, 10
_MIN_CONF = 0.3
WINDOW_HALF = 6  # ±6 frames ≈ 500 ms at 25 fps (paper: 12-frame window)


@dataclass
class FrameState:
    """Per-frame detections needed for assignment: on-court players + the ball."""

    poses: list[PoseDetection]
    ball: BallDetection | None


def _player_distance(pose: PoseDetection, ball_xy: tuple[float, float]) -> float | None:
    """Ball→player distance: min of both wrists if available, else bbox centre."""
    kp = pose.keypoints
    dists = [
        float(np.hypot(kp[w, 0] - ball_xy[0], kp[w, 1] - ball_xy[1]))
        for w in (_L_WRIST, _R_WRIST)
        if kp[w, 2] >= _MIN_CONF
    ]
    if dists:
        return min(dists)
    x1, y1, x2, y2 = pose.bbox_xyxy  # fallback: bbox centre
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return float(np.hypot(cx - ball_xy[0], cy - ball_xy[1]))


def assign_hit(
    hit_frame: int,
    states: dict[int, FrameState],
    window_half: int = WINDOW_HALF,
) -> int | None:
    """Return the player_id that most likely hit the ball at `hit_frame`, or None.

    Weighted majority vote over [hit_frame ± window_half]. Per frame, each player's
    weight is 1 - standardized distance (eq.1 of the paper): the closest player
    gets the most weight, and a whole frame is skipped if it lacks a ball.
    """
    votes: dict[int, float] = {}
    best_overall: dict[int, float] = {}  # player -> smallest distance seen (tie-break)
    for f in range(hit_frame - window_half, hit_frame + window_half + 1):
        st = states.get(f)
        if st is None or st.ball is None or not st.poses:
            continue
        ball_xy = st.ball.image_xy
        dists: dict[int, float] = {}
        for pose in st.poses:
            if pose.player_id is None:
                continue
            d = _player_distance(pose, ball_xy)
            if d is not None:
                dists[pose.player_id] = d
                best_overall[pose.player_id] = min(best_overall.get(pose.player_id, d), d)
        if len(dists) < 2:  # need at least two players to compare
            if dists:  # single player still gets a small vote
                pid = next(iter(dists))
                votes[pid] = votes.get(pid, 0.0) + 1.0
            continue
        dmin, dmax = min(dists.values()), max(dists.values())
        span = (dmax - dmin) or 1.0
        for pid, d in dists.items():
            # eq.1: closer -> higher weight, in [~0.1, 1]
            w = 1.0 - (d - dmin + 0.1) / span
            votes[pid] = votes.get(pid, 0.0) + max(w, 0.0)
    if not votes:
        return None
    top = max(votes.values())
    winners = [pid for pid, v in votes.items() if abs(v - top) < 1e-9]
    if len(winners) == 1:
        return winners[0]
    # tie-break: smallest distance ever seen among the tied players
    return min(winners, key=lambda pid: best_overall.get(pid, float("inf")))


def assign_hits(
    hit_frames: Sequence[int], states: dict[int, FrameState], window_half: int = WINDOW_HALF
) -> dict[int, int | None]:
    """Assign every detected hit frame to a player."""
    return {hf: assign_hit(hf, states, window_half) for hf in hit_frames}

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
MIN_WINDOW_FRAMES = 12  # paper pads short predicted windows up to 500 ms


@dataclass
class FrameState:
    """Per-frame detections needed for assignment: on-court players + the ball."""

    poses: list[PoseDetection]
    ball: BallDetection | None


def _player_distance(pose: PoseDetection, ball_xy: tuple[float, float]) -> float | None:
    """Ball→player distance in BODY HEIGHTS: min of both wrists, else bbox centre.

    Not raw pixels. A far-side player is drawn small, so the same pixel gap means
    a far larger real distance for him than for a near player — and during a
    smash, with the ball high in frame, that bias hands the hit to whoever stands
    at the back. Dividing by the player's own height makes the two comparable.

    Measured against the paper's ground truth (319 hits): smashes go from 60.5%
    to 73.7% correct and the overall figure from 75.5% to 78.4%, with no class
    getting worse. A court-side filter using the ball's projected depth was tried
    first and measured worse (65.2%): a ball in the air does not lie on the
    ground plane, so projecting it invents a position across the net.
    """
    kp = pose.keypoints
    dists = [
        float(np.hypot(kp[w, 0] - ball_xy[0], kp[w, 1] - ball_xy[1]))
        for w in (_L_WRIST, _R_WRIST)
        if kp[w, 2] >= _MIN_CONF
    ]
    if not dists:
        x1, y1, x2, y2 = pose.bbox_xyxy  # fallback: bbox centre
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dists = [float(np.hypot(cx - ball_xy[0], cy - ball_xy[1]))]
    height = max(pose.bbox_xyxy[3] - pose.bbox_xyxy[1], 1.0)
    return min(dists) / height


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


def _team_of(slot: int | None) -> int | None:
    """Players 1-2 are team 1, players 3-4 team 2."""
    return None if slot is None else (1 if slot <= 2 else 2)


def team_alternation_sweep(assignments: dict[int, int | None]) -> dict[int, int | None]:
    """Fill unassigned hits using that teams alternate (paper §5.3, secondary sweep).

    A rally is by definition a sequence of alternating teams: if the hits either
    side of a gap belong to the same team, the missing one is the other team's.
    We can recover the TEAM this way, not the individual player, so the gap is
    filled with a negative marker (-team) meaning "team known, player unknown".
    """
    frames = sorted(assignments)
    out = dict(assignments)
    for i, frame in enumerate(frames):
        if out[frame] is not None:
            continue
        prev_team = next(
            (_team_of(out[frames[j]]) for j in range(i - 1, -1, -1) if out[frames[j]] is not None),
            None,
        )
        next_team = next(
            (
                _team_of(out[frames[j]])
                for j in range(i + 1, len(frames))
                if out[frames[j]] is not None
            ),
            None,
        )
        inferred: int | None = None
        if prev_team is not None and next_team is not None and prev_team == next_team:
            inferred = 2 if prev_team == 1 else 1  # sandwiched: must be the other team
        elif prev_team is not None and next_team is None:
            inferred = 2 if prev_team == 1 else 1
        elif next_team is not None and prev_team is None:
            inferred = 2 if next_team == 1 else 1
        if inferred is not None:
            out[frame] = -inferred  # team-only result
    return out


def frame_window(
    start_s: float, end_s: float, fps: float, min_frames: int = MIN_WINDOW_FRAMES
) -> tuple[int, int]:
    """Video-frame range for a predicted hit window (paper §5.3).

    The model's own onset/offset are used when the window is long enough;
    shorter ones are padded symmetrically to 500 ms (12 frames at 25 fps).
    """
    first, last = round(start_s * fps), round(end_s * fps)
    span = last - first + 1
    if span < min_frames:
        pad = (min_frames - span) / 2
        first, last = round(first - pad), round(last + pad)
    return first, last


def assign_hit_window(
    first_frame: int, last_frame: int, states: dict[int, FrameState]
) -> int | None:
    """Weighted vote over an explicit frame range (see `assign_hit`)."""
    centre = (first_frame + last_frame) // 2
    half = max((last_frame - first_frame) // 2, 1)
    return assign_hit(centre, states, window_half=half)


def assign_hits(
    hit_frames: Sequence[int],
    states: dict[int, FrameState],
    window_half: int = WINDOW_HALF,
    alternation_sweep: bool = True,
) -> dict[int, int | None]:
    """Assign every detected hit frame to a player (negative = team only)."""
    out: dict[int, int | None] = {hf: assign_hit(hf, states, window_half) for hf in hit_frames}
    return team_alternation_sweep(out) if alternation_sweep else out


def assign_hit_windows(
    windows_s: Sequence[tuple[float, float]],
    states: dict[int, FrameState],
    fps: float,
    alternation_sweep: bool = True,
) -> dict[int, int | None]:
    """Assign predicted hit windows (seconds) to players; keyed by centre frame."""
    out: dict[int, int | None] = {}
    for start_s, end_s in windows_s:
        first, last = frame_window(start_s, end_s, fps)
        out[(first + last) // 2] = assign_hit_window(first, last, states)
    return team_alternation_sweep(out) if alternation_sweep else out

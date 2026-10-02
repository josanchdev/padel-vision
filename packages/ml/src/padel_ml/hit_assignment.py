"""Assign a detected hit to the player who made it (ADR-0015, replica of Decorte 2024).

The audio detector says WHEN a hit happens; this says WHO. Replicates the paper's
method (accuracy 83.7% player / 86.8% team), not a naive 1-frame nearest-wrist:

- take a window around the hit (the paper's 500 ms; ours is ±4 frames, see
  `WINDOW_HALF` — measured better on their own ground truth),
- per frame with a ball detection, measure ball→player distance (min of both
  wrists if pose present, else bbox centre), in body heights (`player_distance`),
- weighted majority vote across the window, weight by standardized distance (eq.1),
- tie-break by smallest euclidean distance over all frames.

Multi-frame voting is robust to the frames where pose or ball is momentarily
missing — the single-frame version we tried before was fragile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from padel_cv.pipeline import BallDetection, PoseDetection

_L_WRIST, _R_WRIST = 9, 10
_MIN_CONF = 0.3

WINDOW_HALF = 4
"""Half-width of the voting window, in frames (±160 ms at 25 fps).

The paper uses ±6 (a 500 ms window) and that is what this replicated first.
Swept against its own 319-hit ground truth with the system's configuration
(docs/metrics/assignment_window_sweep.json), narrower is better:

    ±4  87.46% player  92.16% team
    ±5  86.83%         91.54%
    ±6  85.58%         90.60%   <- the paper's width

Frames far from contact have the ball nowhere near the hitter — after it, the
ball is already crossing to the far court and votes for whoever is now closest.

Asymmetry was tested too, since the ball leaves after contact: it does not help
(±6 before-only 82.45%, after-only 81.19%). The window was too wide on both
sides, not mistimed. Shifting the centre also does not help — the optimum sits at
offset -1 frame (+0.6 points), i.e. no audio/video lag worth correcting.
"""


@dataclass
class FrameState:
    """Per-frame detections needed for assignment: on-court players + the ball."""

    poses: list[PoseDetection]
    ball: BallDetection | None


def frame_states(
    players: dict[int, list[PoseDetection]], ball: dict[int, tuple[float, float]]
) -> dict[int, FrameState]:
    """What the vote reads: each frame's identified players and the ball, if seen.

    The one place these are put together, for the system and for every
    measurement of it alike.
    """
    return {
        frame: FrameState(
            poses=poses, ball=BallDetection(ball[frame], 1.0) if frame in ball else None
        )
        for frame, poses in players.items()
    }


def player_distance(pose: PoseDetection, ball_xy: tuple[float, float]) -> float | None:
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


MAX_POSE_GAP = 4
"""Frames a player may be missing before the vote gives up on him.

The detector drops a player for a frame or two — and the hit frame is as likely
as any other to be one of them. Measured over the paper's ground truth, the true
hitter is absent from his own hit frame in 10.0% of hits, but bridging gaps of
up to 4 frames brings that to 8.8%: those were blinks, not absences. Beyond ~4
frames the player really is gone (out of shot, chasing a ball behind the glass)
and holding his last position would invent data.
"""


def _bridge_gaps(
    states: dict[int, FrameState], centre: int, window_half: int, max_gap: int = MAX_POSE_GAP
) -> dict[int, list[PoseDetection]]:
    """Per frame of the window, the poses present plus any briefly-missing player.

    A player who blinks out is carried forward from his nearest sighting, so a
    one-frame dropout at the moment of contact does not cost the whole vote.
    """
    low, high = centre - window_half, centre + window_half
    seen: dict[int, list[tuple[int, PoseDetection]]] = {}
    for frame in range(low - max_gap, high + max_gap + 1):
        st = states.get(frame)
        if st is None:
            continue
        for pose in st.poses:
            if pose.player_id is not None:
                seen.setdefault(pose.player_id, []).append((frame, pose))

    out: dict[int, list[PoseDetection]] = {}
    for frame in range(low, high + 1):
        st = states.get(frame)
        poses = list(st.poses) if st is not None else []
        here = {p.player_id for p in poses}
        for player_id, sightings in seen.items():
            if player_id in here:
                continue
            nearest = min(sightings, key=lambda s: abs(s[0] - frame))
            if abs(nearest[0] - frame) <= max_gap:
                poses.append(nearest[1])
        out[frame] = poses
    return out


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
    bridged = _bridge_gaps(states, hit_frame, window_half)
    for f in range(hit_frame - window_half, hit_frame + window_half + 1):
        st = states.get(f)
        poses = bridged.get(f, [])
        if st is None or st.ball is None or not poses:
            continue
        ball_xy = st.ball.image_xy
        dists: dict[int, float] = {}
        for pose in poses:
            if pose.player_id is None:
                continue
            d = player_distance(pose, ball_xy)
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

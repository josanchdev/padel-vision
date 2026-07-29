"""Detect ball events (shots and bounces) from the ball's trajectory + poses.

A ball event is where the ball's motion sharply CHANGES DIRECTION. Physically:
- a **shot** if a player's wrist is near the ball at that instant (someone hit it)
- a **bounce** if no player is near (it hit the floor or a wall)

This unifies what used to be two heuristics (wrist-speed peaks for shots,
vertical valleys for bounces) into one signal (ADR-0012): find every direction
change, then split them by "is a player there?". More robust than the wrist-speed
peak, which fired on any brisk arm gesture whether or not the ball was struck.

Works in image pixels; the ball comes from our own TrackNet detector, so nothing
here depends on external annotation (ADR-0005).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from padel_cv.bounces import BallSample

# Given a frame index and the ball's pixel position, returns the id of the
# player whose wrist is near the ball (the hitter), or None if none is close.
WristNear = Callable[[int, tuple[float, float]], "int | None"]

# Reuse BallSample (frame_index, x_px, y_px) as the trajectory sample.
__all__ = ["BallEvent", "BallSample", "classify_events", "detect_direction_changes"]


@dataclass(frozen=True)
class BallEvent:
    """A ball direction-change: a shot (with player) or a bounce (without)."""

    frame_index: int
    x_px: float
    y_px: float
    kind: str  # "shot" | "bounce"
    player_id: int | None  # hitter for shots; None for bounces
    turn_deg: float  # how sharply the ball turned (degrees)


def _velocity(a: BallSample, b: BallSample) -> tuple[float, float]:
    dt = b.frame_index - a.frame_index
    if dt == 0:
        return 0.0, 0.0
    return (b.x_px - a.x_px) / dt, (b.y_px - a.y_px) / dt


def _turn_angle(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Angle (deg) between two velocity vectors; 0 = straight, 180 = reversed."""
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def detect_direction_changes(
    samples: list[BallSample],
    window: int = 2,
    min_turn_deg: float = 45.0,
    min_speed_px: float = 3.0,
) -> list[tuple[int, float]]:
    """Return (index, turn_degrees) where the ball changes direction sharply.

    At each sample the incoming velocity (from `window` frames back) is compared
    with the outgoing (to `window` frames ahead). A large turn that isn't just
    slow jitter (both sides moving > min_speed_px) is an event candidate. Only
    the local peak of a turn is kept, so one impact yields one event.
    """
    turns: list[float] = [0.0] * len(samples)
    for i in range(window, len(samples) - window):
        v_in = _velocity(samples[i - window], samples[i])
        v_out = _velocity(samples[i], samples[i + window])
        if math.hypot(*v_in) < min_speed_px or math.hypot(*v_out) < min_speed_px:
            continue
        turns[i] = _turn_angle(v_in, v_out)

    events: list[tuple[int, float]] = []
    for i in range(window, len(samples) - window):
        if turns[i] < min_turn_deg:
            continue
        # Keep only the local maximum of the turn (one event per impact).
        lo, hi = max(0, i - window), min(len(samples), i + window + 1)
        if turns[i] >= max(turns[j] for j in range(lo, hi)):
            events.append((i, turns[i]))
    return events


def classify_events(
    samples: list[BallSample],
    events: list[tuple[int, float]],
    wrist_near: WristNear,
    guard: int = 3,
) -> list[BallEvent]:
    """Split direction-change events into shots (player near) vs bounces.

    wrist_near(frame_index, ball_xy) -> player_id | None: returns the hitter if a
    player's wrist is close to the ball at that frame, else None. Consecutive
    events within `guard` frames collapse to the strongest turn (avoids double
    counting one noisy impact).
    """
    out: list[BallEvent] = []
    for i, turn in sorted(events, key=lambda e: -e[1]):  # strongest first
        s = samples[i]
        if any(abs(s.frame_index - e.frame_index) <= guard for e in out):
            continue  # already have a stronger event near this frame
        player = wrist_near(s.frame_index, (s.x_px, s.y_px))
        out.append(
            BallEvent(
                frame_index=s.frame_index,
                x_px=s.x_px,
                y_px=s.y_px,
                kind="shot" if player is not None else "bounce",
                player_id=player,
                turn_deg=turn,
            )
        )
    return sorted(out, key=lambda e: e.frame_index)

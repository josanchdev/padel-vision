"""Bounce detection from the ball's image trajectory (ADR-0009, Decision F).

A bounce is where the ball hits the floor. With a monocular camera we have no
depth, but we do have the ball's vertical position in image pixels over time.
When the ball bounces, its vertical motion REVERSES: it was descending (y
growing, since image y points down) and starts rising. That local MINIMUM in
height — i.e. a local MAXIMUM of the pixel-y series — is a bounce candidate.

Not every direction change is a floor bounce, though: a player HITTING the ball
also reverses it. We already know when players hit (the pipeline's shot events),
so bounce candidates that coincide with a shot are discarded. What remains are
bounces (floor or wall). Distinguishing floor from wall is a later refinement
(see docs/backlog.md).

Working in image pixels (not projected metres) is deliberate: the homography
maps the floor plane, so a ball in the air projects with error that grows with
its height and would distort the very valley we look for. The homography is used
only to place a confirmed bounce on the court for stats.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallSample:
    """One observation of the ball along its trajectory."""

    frame_index: int
    x_px: float
    y_px: float


@dataclass(frozen=True)
class Bounce:
    """A detected bounce: when and where (image pixels)."""

    frame_index: int
    x_px: float
    y_px: float


def _is_local_max_y(samples: list[BallSample], i: int, window: int) -> bool:
    """True if sample i has the largest y (lowest point) in +-window neighbours.

    A strict-enough local maximum of pixel-y marks the bottom of the ball's arc,
    where descending motion turns to rising — a bounce.
    """
    lo = max(0, i - window)
    hi = min(len(samples), i + window + 1)
    yi = samples[i].y_px
    for j in range(lo, hi):
        if j == i:
            continue
        if samples[j].y_px >= yi:
            return False
    return True


def detect_bounces(
    samples: list[BallSample],
    shot_frames: set[int],
    window: int = 3,
    min_prominence_px: float = 4.0,
    shot_guard: int = 5,
) -> list[Bounce]:
    """Find bounces as local vertical minima not explained by a player's shot.

    - window: half-width (in samples) for the local-maximum test.
    - min_prominence_px: the valley must dip at least this far below the higher
      of its two surrounding peaks, to reject jitter on a near-flat path.
    - shot_guard: candidates within this many frames of a shot are dropped (the
      reversal is the player hitting, not a floor bounce).
    """
    bounces: list[Bounce] = []
    for i in range(len(samples)):
        if not _is_local_max_y(samples, i, window):
            continue
        if _prominence(samples, i, window) < min_prominence_px:
            continue
        frame = samples[i].frame_index
        if any(abs(frame - s) <= shot_guard for s in shot_frames):
            continue
        bounces.append(Bounce(frame, samples[i].x_px, samples[i].y_px))
    return bounces


def _prominence(samples: list[BallSample], i: int, window: int) -> float:
    """How far the valley at i dips below the shallower neighbouring rise.

    Uses the min y (highest point) on each side within the window; the valley's
    prominence is its y minus the larger of those two — small for noise on a
    flat path, large for a real arc bottom.
    """
    lo = max(0, i - window)
    hi = min(len(samples), i + window + 1)
    left = min((samples[j].y_px for j in range(lo, i)), default=samples[i].y_px)
    right = min((samples[j].y_px for j in range(i + 1, hi)), default=samples[i].y_px)
    return samples[i].y_px - max(left, right)

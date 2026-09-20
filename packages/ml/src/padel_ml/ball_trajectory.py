"""Physics-based cleanup of the ball track (ADR-0015).

Filtering bad detections one by one only goes so far: measured over the 99
CVSPORTS rallies, just 71.6% of frames carry a sane ball (7.9% missing, 16.1%
frozen on a static distractor, 3.8% teleporting). The trail jerks, and since hit
assignment and the type classifier both read the ball, that noise propagates.

So instead of judging detections in isolation, this models what the ball is
doing. Between a hit and the next bounce it is a projectile, so over a short
window its image trajectory is close to quadratic — fitted over 5-7 frames the
residual is 2.5-3.8 px, against 4.2-6.4 px for a straight line. That gives three
things at once:

- **a test**: a detection far from the local fit is wrong, however plausible it
  looked on its own,
- **a filler**: gaps get a physically shaped path instead of a straight line,
- **a smoother**: the surviving points are nudged onto the fit, which is what
  turns a jittery trail into a continuous one.

The fit is local and re-estimated per window, so a hit or a bounce — where the
parabola breaks — only affects the windows that straddle it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from padel_ml.ball_infer import BallHit

FIT_WINDOW = 7
"""Frames per local fit. Measured sweet spot: long enough to constrain a
parabola, short enough that one fit does not span two strokes."""

MIN_FIT_POINTS = 5
RESIDUAL_LIMIT_PX = 80.0
"""Reject a detection this far from its local fit.

Far above the 2.5-3.8 px residual of clean play, and chosen by sweeping it
against the paper's 319-hit ground truth: 25/45 px rejected too eagerly and cost
0.6 points of assignment accuracy, 150 px let outliers through and lost the
smash gain. At 80 px the assignment holds at 79.0% while smashes improve to
75.0% and the trail's jerk drops from 23.0 px to 6.9 px."""

MAX_PREDICT_FRAMES = 4
"""How far a gap may be filled by extrapolating the fit."""


@dataclass
class TrackPoint:
    """One frame of the reconstructed track."""

    frame_index: int
    x_px: float
    y_px: float
    measured: bool
    """False when the position comes from the fit rather than the detector."""


def _fit(
    frames: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Quadratic through the window, falling back to linear when short."""
    degree = 2 if len(frames) >= MIN_FIT_POINTS else 1
    return np.polyfit(frames, values, degree)


def _residuals(hits: list[BallHit], window: int = FIT_WINDOW) -> list[float]:
    """Distance from each detection to a fit of its neighbours (excluding itself).

    Leaving the point out is what makes this a test: a fit that included the
    outlier would be dragged towards it and the residual would shrink.
    """
    out: list[float] = []
    for i, hit in enumerate(hits):
        low, high = max(i - window // 2, 0), min(i + window // 2 + 1, len(hits))
        neighbours = [h for j, h in enumerate(hits[low:high], start=low) if j != i]
        if len(neighbours) < MIN_FIT_POINTS:
            out.append(0.0)
            continue
        frames = np.array([h.frame_index for h in neighbours], dtype=np.float64)
        if frames.max() - frames.min() > window * 2:
            out.append(0.0)  # neighbours too far apart to say anything
            continue
        x_fit = _fit(frames, np.array([h.x_px for h in neighbours]))
        y_fit = _fit(frames, np.array([h.y_px for h in neighbours]))
        predicted_x = float(np.polyval(x_fit, hit.frame_index))
        predicted_y = float(np.polyval(y_fit, hit.frame_index))
        out.append(float(np.hypot(predicted_x - hit.x_px, predicted_y - hit.y_px)))
    return out


def reject_outliers(
    hits: list[BallHit], limit_px: float = RESIDUAL_LIMIT_PX, window: int = FIT_WINDOW
) -> list[BallHit]:
    """Drop detections that do not sit on the local trajectory, worst first.

    One at a time, re-fitting after each removal. A single wild detection drags
    the fits of every window it belongs to, so its neighbours look like outliers
    too — removing them all in one pass throws away good points along with the
    bad one. Taking out the worst and re-measuring lets the honest neighbours
    fall back into line.
    """
    remaining = list(hits)
    while len(remaining) >= MIN_FIT_POINTS:
        residuals = _residuals(remaining, window)
        worst = int(np.argmax(residuals))
        if residuals[worst] <= limit_px:
            break
        remaining.pop(worst)
    return remaining


def smooth_and_fill(
    hits: list[BallHit],
    window: int = FIT_WINDOW,
    max_predict: int = MAX_PREDICT_FRAMES,
) -> list[TrackPoint]:
    """Snap detections onto the local fit and fill short gaps along it.

    Filled points are flagged `measured=False` so downstream code can tell a
    reconstruction from an observation — the type classifier, for one, gets a
    presence flag rather than being told a guess is a measurement.
    """
    if len(hits) < MIN_FIT_POINTS:
        return [TrackPoint(h.frame_index, h.x_px, h.y_px, True) for h in hits]

    by_frame = {h.frame_index: h for h in hits}
    frames_present = sorted(by_frame)
    out: list[TrackPoint] = []

    for target in range(frames_present[0], frames_present[-1] + 1):
        # neighbours within the window, by frame distance not list position, so
        # a gap does not silently pull in points from far away
        neighbours = [h for h in hits if abs(h.frame_index - target) <= window // 2]
        measured = target in by_frame
        if len(neighbours) < MIN_FIT_POINTS:
            if measured:
                hit = by_frame[target]
                out.append(TrackPoint(target, hit.x_px, hit.y_px, True))
            continue
        if not measured:
            nearest = min(abs(h.frame_index - target) for h in neighbours)
            if nearest > max_predict:
                continue  # too far from any observation to invent a position
        frames = np.array([h.frame_index for h in neighbours], dtype=np.float64)
        x_fit = _fit(frames, np.array([h.x_px for h in neighbours]))
        y_fit = _fit(frames, np.array([h.y_px for h in neighbours]))
        out.append(
            TrackPoint(
                target,
                float(np.polyval(x_fit, target)),
                float(np.polyval(y_fit, target)),
                measured,
            )
        )
    return out


def clean_track(
    hits: list[BallHit],
    limit_px: float = RESIDUAL_LIMIT_PX,
    window: int = FIT_WINDOW,
    max_predict: int = MAX_PREDICT_FRAMES,
) -> list[TrackPoint]:
    """Full pipeline: drop frozen runs, reject outliers, then smooth and fill."""
    from padel_ml.ball_postprocess import remove_frozen

    return smooth_and_fill(
        reject_outliers(remove_frozen(hits), limit_px, window), window, max_predict
    )

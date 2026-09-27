"""Where the racket met the ball, relative to where the audio heard it.

The audio detector's instant arrives late, for two measured reasons: the
detector marks the peak of the pop rather than its onset, and a broadcast's
sound can lag its picture. The WHO vote looks at +-4 frames around that instant,
so a late instant makes it look at the ball after it has left the racket — the
first external evaluation lost 32 points of player accuracy to exactly this.

The lag is estimated WITHOUT LABELS. At contact the ball is at somebody's
racket, so for each detected hit the frame where the ball comes closest to any
wrist (in body heights, as the vote measures it) marks the contact; the median
gap between detection and contact over many hits is the lag. One scalar per
video, never a per-hit snap: moving each hit to its own closest approach would
let the same ball-to-wrist signal decide both when and who.

Measured over CVSPORTS (production audio model), the lag is 80-120 ms in most
tournaments but 240 ms in AMSTERDAM, and ~170 ms on the external Miami clip —
it belongs to the video, not to the detector alone. Hence two uses of the same
estimate:

    per video   the median over the video's own hits (default)
    global      the median over the training data, for videos with too few
                hits to estimate their own

Validated on VIGO against the paper's annotated instants: the estimated contact
frame falls at a median of 0 ms from them (quartiles -80 / +40 ms).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from padel_cv.pipeline import PoseDetection
from padel_ml.hit_assignment import player_distance

SEARCH_BEFORE_S = 0.5
"""How far before a detection the contact may lie. Wide enough for the worst
lag measured (AMSTERDAM, 240 ms) with margin."""

SEARCH_AFTER_S = 0.25
"""How far after: the detector is rarely early, but allows for it."""

MIN_HITS_PER_VIDEO = 8
"""Below this many usable hits a video's own median is too noisy to trust and
the global lag is used instead."""


@dataclass(frozen=True)
class AudioSync:
    """The lag applied to a video's detections, and where it came from."""

    lag_s: float
    source: str
    """"video" (its own hits), "global" (training data) or "none"."""
    n_hits: int
    """Hits the estimate rests on (0 for "global" and "none")."""


def contact_frame(
    hit_frame: int,
    players: dict[int, list[PoseDetection]],
    ball: dict[int, tuple[float, float]],
    fps: float,
) -> int | None:
    """The frame near a detection where the ball comes closest to a wrist."""
    best: tuple[float, int] | None = None
    first = hit_frame - round(SEARCH_BEFORE_S * fps)
    last = hit_frame + round(SEARCH_AFTER_S * fps)
    for frame in range(first, last + 1):
        position = ball.get(frame)
        poses = players.get(frame)
        if position is None or not poses:
            continue
        distances = [d for p in poses if (d := player_distance(p, position)) is not None]
        if distances and (best is None or min(distances) < best[0]):
            best = (min(distances), frame)
    return None if best is None else best[1]


def lag_samples(
    hit_frames: Sequence[int],
    players: dict[int, list[PoseDetection]],
    ball: dict[int, tuple[float, float]],
    fps: float,
) -> list[float]:
    """Detection minus estimated contact, in seconds, for every usable hit."""
    samples: list[float] = []
    for frame in hit_frames:
        contact = contact_frame(frame, players, ball, fps)
        if contact is not None:
            samples.append((frame - contact) / fps)
    return samples


def choose_sync(samples: Sequence[float], global_lag_s: float) -> AudioSync:
    """The video's own median if it has enough hits, else the global lag."""
    if len(samples) >= MIN_HITS_PER_VIDEO:
        return AudioSync(statistics.median(samples), "video", len(samples))
    return AudioSync(global_lag_s, "global", 0)


def corrected_frames(hit_times_s: Sequence[float], fps: float, lag_s: float) -> list[int]:
    """Detection times moved back by the lag, as sorted unique frames."""
    return sorted({max(round((t - lag_s) * fps), 0) for t in hit_times_s})

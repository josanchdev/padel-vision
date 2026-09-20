"""Clean up raw ball detections (Decorte 2024, §4.3) — see ADR-0015.

TrackNet emits an independent detection per frame, so its output has two flaws
the paper explicitly post-processes away:

- **Teleportation**: an isolated detection metres away from the trajectory,
  usually a bright distractor (a line, a shoe, a logo). A real ball moves
  continuously, so a jump that big in one frame is a false positive.
- **Gaps**: the ball vanishes behind a player or leaves the frame. Short gaps
  are interpolated, which matters because hit assignment needs a ball position
  in as many frames of the hit window as possible.
- **Frozen detections**: the opposite failure, and the costly one here. The
  detector latches onto a static distractor and reports the very same pixel for
  dozens of frames. Measured across the 99 CVSPORTS rallies, 17% of all ball
  detections repeat the previous frame exactly — a real ball never does that.
  Worse, a frozen point sits next to whichever player happens to be there, so
  hit assignment confidently credits every hit to them: this is what put smashes
  on the wrong side of the net.

Long gaps are left alone: interpolating across them would invent a trajectory.
"""

from __future__ import annotations

from itertools import pairwise

from padel_ml.ball_infer import BallHit

MAX_JUMP_PX = 250.0
"""Max plausible ball displacement between consecutive frames (1080p, 25 fps)."""

MAX_GAP_FRAMES = 6
"""Gaps longer than this stay empty rather than being invented."""

MAX_FROZEN_FRAMES = 3
"""A ball repeating the same pixel longer than this is a static false positive.

Three frames is 120 ms at 25 fps. A real ball can look static for a frame or two
at the apex of a lob or through rounding, but not beyond that."""

FROZEN_TOLERANCE_PX = 2.0
"""How close two detections must be to count as the same point."""


def remove_frozen(
    hits: list[BallHit],
    max_frozen: int = MAX_FROZEN_FRAMES,
    tolerance_px: float = FROZEN_TOLERANCE_PX,
) -> list[BallHit]:
    """Drop runs where the detection stops moving — a latched distractor.

    The whole run goes, not just its tail: the first frame of a frozen run is as
    wrong as the last, and keeping it would leave a false anchor exactly where
    the trajectory should have continued.
    """
    if not hits:
        return []
    out: list[BallHit] = []
    run: list[BallHit] = [hits[0]]
    for hit in hits[1:]:
        previous = run[-1]
        same_spot = (
            abs(hit.x_px - previous.x_px) <= tolerance_px
            and abs(hit.y_px - previous.y_px) <= tolerance_px
        )
        if same_spot:
            run.append(hit)
            continue
        if len(run) <= max_frozen:
            out.extend(run)
        run = [hit]
    if len(run) <= max_frozen:
        out.extend(run)
    return out


def remove_teleports(hits: list[BallHit], max_jump_px: float = MAX_JUMP_PX) -> list[BallHit]:
    """Drop detections that jump implausibly far from the established trajectory.

    A detection is rejected when it is far from the previous accepted one AND
    the next detection returns near that previous one — i.e. a one-frame
    excursion, not the ball genuinely moving on.
    """
    if len(hits) < 3:
        return list(hits)
    kept = [hits[0]]
    for i in range(1, len(hits) - 1):
        prev, cur, nxt = kept[-1], hits[i], hits[i + 1]
        span = max(cur.frame_index - prev.frame_index, 1)
        d_prev = float(((cur.x_px - prev.x_px) ** 2 + (cur.y_px - prev.y_px) ** 2) ** 0.5)
        if d_prev <= max_jump_px * span:
            kept.append(cur)
            continue
        # far from the trajectory: keep it only if the next frame follows it
        d_next_from_cur = float(((nxt.x_px - cur.x_px) ** 2 + (nxt.y_px - cur.y_px) ** 2) ** 0.5)
        d_next_from_prev = float(((nxt.x_px - prev.x_px) ** 2 + (nxt.y_px - prev.y_px) ** 2) ** 0.5)
        if d_next_from_cur < d_next_from_prev:
            kept.append(cur)  # the trajectory really did move there
    kept.append(hits[-1])
    return kept


def interpolate_gaps(hits: list[BallHit], max_gap_frames: int = MAX_GAP_FRAMES) -> list[BallHit]:
    """Linearly fill short gaps between detections (confidence 0 = interpolated)."""
    if len(hits) < 2:
        return list(hits)
    out: list[BallHit] = []
    for prev, nxt in pairwise(hits):
        out.append(prev)
        gap = nxt.frame_index - prev.frame_index
        if 1 < gap <= max_gap_frames + 1:
            for step in range(1, gap):
                t = step / gap
                out.append(
                    BallHit(
                        frame_index=prev.frame_index + step,
                        x_px=prev.x_px + t * (nxt.x_px - prev.x_px),
                        y_px=prev.y_px + t * (nxt.y_px - prev.y_px),
                        confidence=0.0,
                    )
                )
    out.append(hits[-1])
    return out


def postprocess_ball(
    hits: list[BallHit],
    max_jump_px: float = MAX_JUMP_PX,
    max_gap_frames: int = MAX_GAP_FRAMES,
    max_frozen: int = MAX_FROZEN_FRAMES,
) -> list[BallHit]:
    """Drop frozen runs and teleports, then interpolate the short gaps left.

    Frozen first: a latched run would otherwise look like a perfectly consistent
    trajectory to the teleport filter, which only questions movement.
    """
    cleaned = remove_teleports(remove_frozen(hits, max_frozen), max_jump_px)
    return interpolate_gaps(cleaned, max_gap_frames)

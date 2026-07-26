"""Turn attributed shots into normalized skeleton clips for training.

For each attributed shot (impact frame + hitter's track id), we follow that
track across a fixed window centered on the impact and normalize each skeleton
so the classifier sees pose *shape over time*, invariant to where on court the
player is, their size, or the camera. Negative ("no-shot") clips are sampled
from rally frames far from any shot so the model learns to reject non-strokes
(the false positives the wrist heuristic produced).

Skeletons here are always OUR YOLO26 poses (ADR-0008): the training data
matches inference, and nothing depends on external annotations at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from padel_cv.pose_cache import PoseCache
from padel_cv.shot_clips import SHOT_CLASSES, AttributedShot, canonical_label

FloatArray = npt.NDArray[np.float32]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
MIN_KEYPOINT_CONFIDENCE = 0.3

WINDOW = 32  # frames per clip (~1 s at 30 fps); wider than the ~17-frame shot
NO_SHOT_LABEL = "NoShot"
CLIP_CLASSES = [*SHOT_CLASSES, NO_SHOT_LABEL]


@dataclass
class Clip:
    keypoints: FloatArray  # (WINDOW, 17, 3): normalized (x, y) + confidence
    label: str
    source_frame: int


def normalize_skeleton(skeleton: FloatArray) -> FloatArray:
    """Center on hip midpoint, scale by torso length. Keeps the confidence col.

    Returns a copy; low-confidence joints keep their (now normalized) position
    but their confidence flags them so a model can down-weight them.
    """
    out = skeleton.copy().astype(np.float32)
    hips = out[[LEFT_HIP, RIGHT_HIP], :2]
    shoulders = out[[LEFT_SHOULDER, RIGHT_SHOULDER], :2]
    hip_center = hips.mean(axis=0)
    torso = float(np.linalg.norm(shoulders.mean(axis=0) - hip_center))
    scale = torso if torso > 1.0 else 1.0
    out[:, :2] = (out[:, :2] - hip_center) / scale
    return out


def _track_window(
    cache: PoseCache, track_id: int, impact_frame: int, window: int
) -> FloatArray | None:
    """Follow a track across a window centered on impact; hold-fill gaps.

    Returns (window, 17, 3) of normalized skeletons, or None if the track is
    missing for more than half the window (too unreliable to keep).
    """
    half = window // 2
    frames = range(impact_frame - half, impact_frame - half + window)
    skeletons: list[FloatArray | None] = []
    for frame_index in frames:
        fp = cache.get(frame_index)
        match = None
        if fp is not None:
            hits = np.where(fp.track_ids == track_id)[0]
            if len(hits):
                match = normalize_skeleton(fp.keypoints[hits[0]])
        skeletons.append(match)

    present = [s for s in skeletons if s is not None]
    if len(present) < window // 2:
        return None
    # Hold-fill: forward-fill from the nearest earlier frame, else backward.
    filled: list[FloatArray] = []
    last: FloatArray | None = None
    for s in skeletons:
        if s is not None:
            last = s
        filled.append(s if s is not None else (last if last is not None else present[0]))
    return np.stack(filled).astype(np.float32)


def build_shot_clips(
    cache: PoseCache, attributed: list[AttributedShot], window: int = WINDOW
) -> list[Clip]:
    """One clip per attributed shot whose track survives the window."""
    clips: list[Clip] = []
    for shot in attributed:
        track_id = _track_id_at(cache, shot)
        if track_id is None:
            continue
        keypoints = _track_window(cache, track_id, shot.impact_frame, window)
        if keypoints is None:
            continue
        clips.append(
            Clip(
                keypoints=keypoints,
                label=canonical_label(shot.run.shot_type),
                source_frame=shot.impact_frame,
            )
        )
    return clips


def _track_id_at(cache: PoseCache, shot: AttributedShot) -> int | None:
    fp = cache.get(shot.impact_frame)
    if fp is None or shot.player_index >= len(fp.track_ids):
        return None
    track_id = int(fp.track_ids[shot.player_index])
    return track_id if track_id >= 0 else None


def sample_no_shot_clips(
    cache: PoseCache,
    shot_frames: set[int],
    n_clips: int,
    window: int = WINDOW,
    min_gap: int = 45,
    seed: int = 0,
) -> list[Clip]:
    """Sample negative clips from tracked players far from any shot frame."""
    rng = np.random.default_rng(seed)
    candidates = [
        f for f in _cached_frames(cache) if all(abs(f - s) > min_gap for s in shot_frames)
    ]
    rng.shuffle(candidates)
    clips: list[Clip] = []
    for frame_index in candidates:
        if len(clips) >= n_clips:
            break
        fp = cache.get(frame_index)
        if fp is None or len(fp.track_ids) == 0:
            continue
        track_id = int(fp.track_ids[rng.integers(len(fp.track_ids))])
        if track_id < 0:
            continue
        keypoints = _track_window(cache, track_id, frame_index, window)
        if keypoints is not None:
            clips.append(Clip(keypoints=keypoints, label=NO_SHOT_LABEL, source_frame=frame_index))
    return clips


def _cached_frames(cache: PoseCache) -> list[int]:
    return sorted(cache.keypoints_by_frame().keys())

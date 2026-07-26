import numpy as np

from padel_cv.clip_builder import (
    CLIP_CLASSES,
    NO_SHOT_LABEL,
    Clip,
    build_shot_clips,
    normalize_skeleton,
    sample_no_shot_clips,
)
from padel_cv.pose_cache import FramePoses, PoseCache
from padel_cv.shot_clips import AttributedShot, ShotRun


def make_skeleton(cx: float, cy: float, torso: float) -> np.ndarray:
    """Upright skeleton at (cx, cy); all offsets scale with torso so the same
    posture at different sizes normalizes identically."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [cx - 0.20 * torso, cy - torso, 0.9]  # left shoulder
    kp[6] = [cx + 0.20 * torso, cy - torso, 0.9]  # right shoulder
    kp[11] = [cx - 0.16 * torso, cy, 0.9]  # left hip
    kp[12] = [cx + 0.16 * torso, cy, 0.9]  # right hip
    kp[10] = [cx + 0.40 * torso, cy - 0.5 * torso, 0.9]  # right wrist
    return kp


class FakeCache(PoseCache):
    def __init__(self, frames: dict[int, FramePoses]) -> None:
        self._frames = frames


def test_normalize_is_translation_and_scale_invariant() -> None:
    near = make_skeleton(500, 800, torso=100)
    far = make_skeleton(1400, 300, torso=40)
    joints = [5, 6, 10, 11, 12]  # the joints the helper actually sets
    # Same posture at different court positions/sizes must normalize alike.
    np.testing.assert_allclose(
        normalize_skeleton(near)[joints, :2], normalize_skeleton(far)[joints, :2], atol=1e-5
    )


def test_normalize_puts_hip_center_at_origin() -> None:
    norm = normalize_skeleton(make_skeleton(500, 800, torso=100))
    hip_center = norm[[11, 12], :2].mean(axis=0)
    np.testing.assert_allclose(hip_center, [0.0, 0.0], atol=1e-6)


def _cache_with_track(track_id: int, frames: range) -> FakeCache:
    data = {}
    for f in frames:
        kp = make_skeleton(500 + f, 800, torso=100)[None]  # (1, 17, 3)
        data[f] = FramePoses(
            keypoints=kp, track_ids=np.array([track_id]), boxes=np.zeros((1, 4), np.float32)
        )
    return FakeCache(data)


def test_build_shot_clip_follows_track_and_labels() -> None:
    cache = _cache_with_track(track_id=3, frames=range(0, 100))
    shot = AttributedShot(
        ShotRun(48, 52, "Dropshot"), impact_frame=50, player_index=0, wrist_ball_dist_px=20.0
    )
    clips = build_shot_clips(cache, [shot], window=32)
    assert len(clips) == 1
    assert clips[0].keypoints.shape == (32, 17, 3)
    assert clips[0].label == "Other"  # Dropshot folds into Other


def test_build_shot_clip_dropped_when_track_absent() -> None:
    cache = _cache_with_track(track_id=3, frames=range(0, 100))
    # Attribution points at a track id that does not exist at impact.
    bad = FramePoses(
        keypoints=make_skeleton(500, 800, 100)[None],
        track_ids=np.array([99]),
        boxes=np.zeros((1, 4), np.float32),
    )
    cache._frames[50] = bad
    shot = AttributedShot(ShotRun(50, 50, "Smash"), 50, player_index=0, wrist_ball_dist_px=20.0)
    # track 99 exists only at frame 50 -> window mostly empty -> dropped
    assert build_shot_clips(cache, [shot], window=32) == []


def test_no_shot_clips_avoid_shot_frames() -> None:
    cache = _cache_with_track(track_id=3, frames=range(0, 400))
    clips = sample_no_shot_clips(cache, shot_frames={200}, n_clips=5, window=32, min_gap=45)
    assert 0 < len(clips) <= 5
    assert all(c.label == NO_SHOT_LABEL for c in clips)
    assert all(abs(c.source_frame - 200) > 45 for c in clips)


def test_clip_classes_include_no_shot() -> None:
    assert NO_SHOT_LABEL in CLIP_CLASSES
    assert len(CLIP_CLASSES) == 6
    assert isinstance(Clip(np.zeros((32, 17, 3), np.float32), "Smash", 10).source_frame, int)

import numpy as np
from padel_ml.audio_sync import (
    MIN_HITS_PER_VIDEO,
    choose_sync,
    contact_frame,
    corrected_frames,
    lag_samples,
)

from padel_cv.pipeline import PoseDetection

FPS = 25.0


def _player(player_id: int, cx: float, cy: float) -> PoseDetection:
    """A 200 px tall player whose right wrist sits at (cx + 40, cy - 40)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[:, 0], kp[:, 1] = cx, cy
    kp[9] = (cx - 40, cy - 40, 1)
    kp[10] = (cx + 40, cy - 40, 1)
    return PoseDetection(
        bbox_xyxy=(cx - 50, cy - 100, cx + 50, cy + 100),
        confidence=1.0,
        keypoints=kp,
        player_id=player_id,
    )


def _rally(contacts: list[int], n_frames: int = 400):  # type: ignore[no-untyped-def]
    """One player; the ball touches his wrist at each contact frame and is far
    away the rest of the time (it grows farther the further from contact)."""
    players = {i: [_player(1, 500, 500)] for i in range(n_frames)}
    wrist = np.array([540.0, 460.0])
    ball = {}
    for i in range(n_frames):
        gap = min(abs(i - c) for c in contacts)
        x, y = wrist[0] + 30.0 * gap, wrist[1] - 20.0 * gap
        ball[i] = (float(x), float(y))
    return players, ball


def test_contact_is_where_the_ball_meets_a_wrist() -> None:
    players, ball = _rally([100])
    assert contact_frame(104, players, ball, FPS) == 100  # detection 4 frames late


def test_lag_is_detection_minus_contact_in_seconds() -> None:
    contacts = [50, 120, 190, 260]
    players, ball = _rally(contacts)
    samples = lag_samples([c + 3 for c in contacts], players, ball, FPS)
    assert samples == [3 / FPS] * 4


def test_a_video_with_enough_hits_uses_its_own_lag() -> None:
    sync = choose_sync([0.12] * MIN_HITS_PER_VIDEO, global_lag_s=0.08)
    assert sync.source == "video" and sync.lag_s == 0.12


def test_too_few_hits_fall_back_to_the_global_lag() -> None:
    sync = choose_sync([0.12] * (MIN_HITS_PER_VIDEO - 1), global_lag_s=0.08)
    assert sync.source == "global" and sync.lag_s == 0.08


def test_median_ignores_a_stray_estimate() -> None:
    """One hit locked onto the wrong moment must not drag the whole video."""
    samples = [0.12] * (MIN_HITS_PER_VIDEO - 1) + [0.9]
    assert choose_sync(samples, global_lag_s=0.0).lag_s == 0.12


def test_corrected_frames_move_back_by_the_lag() -> None:
    assert corrected_frames([4.0, 8.0], FPS, lag_s=0.12) == [97, 197]
    assert corrected_frames([0.02], FPS, lag_s=0.12) == [0]  # never before the video

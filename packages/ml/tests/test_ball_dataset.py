import numpy as np
import torch
from padel_ml.ball_dataset import GRID_H, GRID_W, BallClips, _consecutive_windows

from padel_cv.ball_cache import FRAME_H, FRAME_W


def _write_shard(cache_dir, indices, centers, occluded=None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(indices)
    frames = np.zeros((n, FRAME_H, FRAME_W, 3), dtype=np.uint8)
    # Tag each frame with its index in pixel [0,0] so window ordering is checkable.
    for k, idx in enumerate(indices):
        frames[k, 0, 0, 0] = idx
    if occluded is None:
        occluded = [False] * n
    np.savez_compressed(
        cache_dir / f"frames_{indices[0]:07d}.npz",
        frames=frames,
        indices=np.asarray(indices, dtype=np.int64),
        centers=np.asarray(centers, dtype=np.float32),
        occluded=np.asarray(occluded, dtype=np.bool_),
    )


def test_consecutive_windows_skips_gaps() -> None:
    # Frames 0,1,2 consecutive; then a gap to 5,6,7.
    indices = np.array([0, 1, 2, 5, 6, 7])
    # Valid window ends: position 2 (frames 0-1-2) and position 5 (frames 5-6-7).
    assert _consecutive_windows(indices) == [2, 5]


def test_neg_ratio_subsamples_ball_absent_windows(tmp_path) -> None:
    cache = tmp_path / "match"
    # 10 consecutive frames: 2 with a ball (targets), 8 without.
    nan = (np.nan, np.nan)
    centers = [(0.5, 0.5), (0.5, 0.5)] + [nan] * 8  # windows end at pos 2..9
    _write_shard(cache, indices=list(range(10)), centers=centers)
    # Windows end at positions 2..9 (8 windows). Positives: end at 2 (ball) only,
    # since ball is at frames 0,1 and window target is the LAST frame.
    full = BallClips([cache])
    balanced = BallClips([cache], neg_ratio=1.0)
    assert len(balanced) < len(full)
    # With 1 positive window, neg_ratio=1.0 keeps at most 1 negative -> 2 total.
    positives = sum(1 for i in range(len(full)) if full[i][1].max() > 0)
    assert len(balanced) <= 2 * max(positives, 1)


def test_dataset_stacks_three_frames_and_targets_last(tmp_path) -> None:
    cache = tmp_path / "match"
    _write_shard(
        cache,
        indices=[0, 1, 2, 3],
        centers=[(0.5, 0.5), (np.nan, np.nan), (0.25, 0.25), (0.5, 0.5)],
    )
    ds = BallClips([cache])
    # 4 consecutive frames -> 2 windows (ending at 2 and 3).
    assert len(ds) == 2

    x, y, occ = ds[0]
    assert x.shape == (9, FRAME_H, FRAME_W)  # 3 frames x 3 channels
    assert y.shape == (GRID_H, GRID_W)
    assert occ.item() is False
    # Window 0 targets frame 2 (center 0.25,0.25) -> non-empty heatmap.
    assert y.max() > 0.5

    # The first window's three frames are 0,1,2 in order (tag lives at [0,0,0]
    # of each frame; channels are frame-major so channels 0,3,6 hold them).
    assert x[0, 0, 0].item() == 0.0  # frame 0 tag / 255
    assert torch.isclose(x[3, 0, 0], torch.tensor(1.0 / 255.0))  # frame 1
    assert torch.isclose(x[6, 0, 0], torch.tensor(2.0 / 255.0))  # frame 2


def test_dataset_absent_ball_gives_zero_target(tmp_path) -> None:
    cache = tmp_path / "match"
    _write_shard(
        cache,
        indices=[0, 1, 2],
        centers=[(0.5, 0.5), (0.5, 0.5), (np.nan, np.nan)],
    )
    ds = BallClips([cache])
    _, y, _ = ds[0]  # targets frame 2, which has no ball
    assert not y.any()


def test_dataset_carries_occluded_flag_of_target_frame(tmp_path) -> None:
    cache = tmp_path / "match"
    _write_shard(
        cache,
        indices=[0, 1, 2],
        centers=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
        occluded=[False, False, True],  # target frame (2) is occluded
    )
    ds = BallClips([cache])
    _, _, occ = ds[0]
    assert occ.item() is True

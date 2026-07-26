import numpy as np

from padel_cv.pose_cache import PoseCache, _covered_frames, _shard_path


def write_shard(cache_dir, start, frame_indices, counts, n_total):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _shard_path(cache_dir, start),
        frames=np.array(frame_indices, dtype=np.int64),
        counts=np.array(counts, dtype=np.int64),
        keypoints=np.zeros((n_total, 17, 3), np.float32),
        track_ids=np.arange(n_total, dtype=np.int64),
        boxes=np.zeros((n_total, 4), np.float32),
    )


def test_covered_frames_counts_contiguous_prefix(tmp_path):
    write_shard(tmp_path, 0, [0, 1, 2], [1, 1, 1], 3)
    write_shard(tmp_path, 3, [3, 4], [1, 1], 2)
    assert _covered_frames(tmp_path, shard_size=5) == 5


def test_covered_frames_stops_at_gap(tmp_path):
    write_shard(tmp_path, 0, [0, 1], [1, 1], 2)
    write_shard(tmp_path, 10, [10, 11], [1, 1], 2)  # gap: frames 2-9 missing
    assert _covered_frames(tmp_path, shard_size=5) == 2


def test_covered_frames_handles_partial_final_shard(tmp_path):
    # A shard whose frame count is far below shard_size must not be treated
    # as if it covered the whole shard window.
    write_shard(tmp_path, 0, [0, 1, 2], [1, 1, 1], 3)  # only 3 of shard_size=5000
    assert _covered_frames(tmp_path, shard_size=5000) == 3


def test_covered_frames_empty(tmp_path):
    assert _covered_frames(tmp_path, shard_size=5) == 0


def test_pose_cache_reconstructs_per_frame_detections(tmp_path):
    # Frame 0 has 2 detections, frame 1 has 1 detection.
    np.savez_compressed(
        _shard_path(tmp_path, 0),
        frames=np.array([0, 1], dtype=np.int64),
        counts=np.array([2, 1], dtype=np.int64),
        keypoints=np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3),
        track_ids=np.array([7, 8, 9], dtype=np.int64),
        boxes=np.zeros((3, 4), np.float32),
    )
    cache = PoseCache(tmp_path)
    assert len(cache) == 2
    frame0 = cache.get(0)
    assert frame0 is not None
    assert frame0.keypoints.shape == (2, 17, 3)
    assert frame0.track_ids.tolist() == [7, 8]
    frame1 = cache.get(1)
    assert frame1 is not None
    assert frame1.track_ids.tolist() == [9]
    assert cache.get(99) is None

import numpy as np
from padel_ml.audio_train import windows_from_frames


def test_finds_contiguous_runs() -> None:
    probs = np.array([0.1, 0.9, 0.95, 0.8, 0.1, 0.1, 0.7, 0.2], dtype=np.float32)
    assert windows_from_frames(probs, threshold=0.5, min_gap_frames=2) == [(1, 3), (6, 6)]


def test_merges_runs_closer_than_min_gap() -> None:
    probs = np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9], dtype=np.float32)
    assert windows_from_frames(probs, threshold=0.5, min_gap_frames=3) == [(0, 2), (6, 6)]


def test_run_reaching_the_end_is_closed() -> None:
    probs = np.array([0.1, 0.9, 0.9], dtype=np.float32)
    assert windows_from_frames(probs, threshold=0.5) == [(1, 2)]

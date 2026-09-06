import numpy as np
from padel_ml.audio_dataset import frame_time
from padel_ml.audio_train import event_eval, peaks_from_frames


def test_peaks_from_frames_finds_maxima() -> None:
    p = np.zeros(100, dtype=np.float32)
    p[20] = 0.9
    p[60] = 0.8
    peaks = peaks_from_frames(p, threshold=0.5, min_gap_frames=4)
    assert peaks == [20, 60]


def test_peaks_respect_min_gap() -> None:
    p = np.zeros(50, dtype=np.float32)
    p[20] = 0.9
    p[22] = 0.7  # too close, suppressed
    assert peaks_from_frames(p, 0.5, min_gap_frames=4) == [20]


def test_event_eval_matches_within_collar() -> None:
    # a real hit centred at frame_time(30); a predicted peak at 30 should match
    t = frame_time(30)
    tp, fp, fn = event_eval([30], [(t - 0.05, t + 0.05)], collar_s=0.25)
    assert (tp, fp, fn) == (1, 0, 0)


def test_event_eval_counts_fp_and_fn() -> None:
    # one real hit far from the only prediction -> 1 FP + 1 FN
    tp, fp, fn = event_eval([10], [(100.0, 100.1)], collar_s=0.25)
    assert (tp, fp, fn) == (0, 1, 1)

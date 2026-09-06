import numpy as np
from padel_ml.audio_dataset import (
    HOP,
    N_FFT,
    SAMPLE_RATE,
    frame_time,
    hit_labels,
    tournament_of,
)


def test_frame_time_increases_by_hop() -> None:
    t0 = frame_time(0)
    t1 = frame_time(1)
    assert abs((t1 - t0) - HOP / SAMPLE_RATE) < 1e-9
    # frame 0 centred at N_FFT/2 samples
    assert abs(t0 - (N_FFT / 2) / SAMPLE_RATE) < 1e-9


def test_hit_labels_mark_window_frames() -> None:
    # a hit from 1.0s to 1.2s should light up the frames in that window
    n = 400
    labels = hit_labels(n, [(1.0, 1.2)])
    times = np.array([frame_time(i) for i in range(n)])
    inside = (times >= 1.0) & (times <= 1.2)
    assert np.array_equal(labels.astype(bool), inside)
    assert labels.sum() > 0


def test_hit_labels_multiple_and_empty() -> None:
    assert hit_labels(100, []).sum() == 0
    two = hit_labels(2000, [(0.5, 0.6), (5.0, 5.1)])
    # two separate positive runs
    changes = np.diff(two)
    assert (changes == 1).sum() == 2  # two rising edges


def test_tournament_of_groups_by_date_location() -> None:
    assert tournament_of("20230528_VIGO_00.mp4") == "20230528_VIGO"
    assert tournament_of("20231112_MALMO_06.mp4") == "20231112_MALMO"

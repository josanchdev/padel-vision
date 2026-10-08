import numpy as np
from padel_ml.thesis.audio_figure import AudioTrace


def test_window_keeps_the_frames_and_events_inside_it() -> None:
    times = np.arange(0.0, 10.0, 0.5, dtype=np.float32)
    trace = AudioTrace(
        times_s=times,
        log_mel=np.zeros((len(times), 40), np.float32),
        probability=np.linspace(0, 1, len(times), dtype=np.float32),
        detected_s=[1.0, 4.2, 9.5],
        annotated=[(0.8, 1.2), (3.9, 4.4), (9.3, 9.8)],
    )
    shown = trace.window(0.0, 5.0)
    assert shown.times_s.min() >= 0.0 and shown.times_s.max() <= 5.0
    assert len(shown.times_s) == len(shown.probability) == shown.log_mel.shape[0]
    assert shown.detected_s == [1.0, 4.2]
    assert shown.annotated == [(0.8, 1.2), (3.9, 4.4)]


def test_window_keeps_an_annotated_hit_that_straddles_its_edge() -> None:
    times = np.arange(0.0, 6.0, 0.5, dtype=np.float32)
    trace = AudioTrace(
        times,
        np.zeros((len(times), 40), np.float32),
        np.zeros(len(times), np.float32),
        [],
        [(4.8, 5.3)],
    )
    assert trace.window(0.0, 5.0).annotated == [(4.8, 5.3)]

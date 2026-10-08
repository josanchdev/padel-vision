import numpy as np
from padel_ml.thesis.audio_figure import AudioTrace, mel_position


def test_window_keeps_the_frames_and_events_inside_it() -> None:
    times = np.arange(0.0, 10.0, 0.5, dtype=np.float32)
    trace = AudioTrace(
        times_s=times,
        log_mel=np.zeros((len(times), 40), np.float32),
        probability=np.linspace(0, 1, len(times), dtype=np.float32),
        detected_s=[1.0, 4.2, 9.5],
        annotated=[(0.8, 1.2), (3.9, 4.4), (9.3, 9.8)],
        sample_rate=48000,
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
        48000,
    )
    assert trace.window(0.0, 5.0).annotated == [(4.8, 5.3)]


def test_mel_position_runs_from_the_lowest_band_to_nyquist() -> None:
    assert mel_position(0.0, 40, 48000) == 0.0
    assert abs(mel_position(24000.0, 40, 48000) - 41.0) < 1e-6
    # 1 kHz is about 1000 mel, a quarter of the way to 24 kHz (about 4016 mel)
    assert 10.0 < mel_position(1000.0, 40, 48000) < 10.5

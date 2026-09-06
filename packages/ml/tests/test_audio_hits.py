import numpy as np
from padel_ml.audio_hits import detect_peaks, onset_envelope


def test_onset_envelope_spikes_on_energy_jump() -> None:
    sr = 22050
    x = np.zeros(sr, dtype=np.float32)  # 1s silence
    # inject a short loud burst at 0.5s (a "hit")
    start = sr // 2
    x[start : start + 200] = 1.0
    env, hop_s = onset_envelope(x, sr)
    # the envelope should peak near 0.5s
    peak_idx = int(np.argmax(env))
    peak_time = peak_idx * hop_s
    assert abs(peak_time - 0.5) < 0.05


def test_detect_peaks_finds_bursts_and_respects_gap() -> None:
    hop_s = 0.005
    env = np.zeros(400, dtype=np.float32)
    env[100] = 0.9  # burst 1 at 0.5s
    env[300] = 0.8  # burst 2 at 1.5s
    hits = detect_peaks(env, hop_s, threshold=0.3, min_gap_s=0.2)
    times = [round(h.time_s, 2) for h in hits]
    assert times == [0.5, 1.5]


def test_detect_peaks_dedups_within_gap() -> None:
    hop_s = 0.005
    env = np.zeros(200, dtype=np.float32)
    env[100] = 0.9
    env[110] = 0.7  # 50ms later -> same hit, must be suppressed
    hits = detect_peaks(env, hop_s, threshold=0.3, min_gap_s=0.2)
    assert len(hits) == 1
    assert hits[0].time_s == 0.5


def test_detect_peaks_below_threshold_empty() -> None:
    env = np.full(100, 0.1, dtype=np.float32)
    assert detect_peaks(env, 0.005, threshold=0.3) == []

import numpy as np
from padel_ml.thesis.spectrograms import (
    Spectrogram,
    clock,
    compute_spectrogram,
    hertz,
    isolated_quiet_stretch,
)


def test_clock_and_hertz_follow_the_figure_axes() -> None:
    assert [clock(s) for s in (0, 15, 75.4, 3600)] == ["0:00", "0:15", "1:15", "60:00"]
    assert hertz(0) == "0" and hertz(12000) == r"12\,000" and hertz(24000) == r"24\,000"


def test_a_click_shows_as_a_stripe_at_its_instant() -> None:
    sample_rate = 48000
    samples = np.zeros(sample_rate, np.float32)
    samples[24000] = 1.0
    spectrogram = compute_spectrogram(samples, sample_rate)
    loudest = int(np.argmax(spectrogram.decibels.max(axis=0)))
    assert abs(loudest * spectrogram.column_s - 0.5) < 0.03
    assert spectrogram.nyquist_hz == 24000


def test_cut_starts_again_from_zero_and_shrunk_keeps_the_loudest() -> None:
    decibels = np.full((4, 1000), -60.0, np.float32)
    decibels[2, 501] = 0.0
    full = Spectrogram(decibels, 0.01, 24000)
    part = full.cut(5.0, 6.0)
    assert part.decibels.shape == (4, 100) and part.duration_s == 1.0
    assert part.decibels.max() == 0.0
    small = full.shrunk(columns=100)
    assert small.decibels.shape == (4, 100) and small.column_s == 0.1
    assert small.decibels.max() == 0.0


def test_the_quiet_stretch_avoids_annotated_hits() -> None:
    energy = np.zeros(1000, np.float32)
    energy[300:400] = 10.0  # the loudest part, but a hit is annotated there
    energy[700:800] = 5.0
    start = isolated_quiet_stretch(energy, 0.01, [(3.2, 3.6)], 1.0, 0.3)
    assert 6.9 <= start <= 7.1

import numpy as np
from padel_ml.thesis.spectrograms import (
    Spectrogram,
    colour,
    compute_spectrogram,
    hertz,
    isolated_quiet_stretch,
)


def test_hertz_follows_the_axis_style() -> None:
    assert hertz(0) == "0" and hertz(12000) == r"12\,000" and hertz(24000) == r"24\,000"


def test_a_click_shows_as_a_stripe_at_its_instant() -> None:
    sample_rate = 48000
    samples = np.zeros(sample_rate, np.float32)
    samples[24000] = 1.0
    spectrogram = compute_spectrogram(samples, sample_rate)
    loudest = int(np.argmax(spectrogram.decibels.max(axis=0)))
    assert abs(loudest * spectrogram.column_s - 0.5) < 0.01
    assert spectrogram.nyquist_hz == 24000


def test_padding_gives_more_frequency_points_than_the_window_has_samples() -> None:
    spectrogram = compute_spectrogram(np.zeros(4800, np.float32), 48000)
    assert spectrogram.decibels.shape[0] > 512


def test_cut_starts_again_from_zero() -> None:
    decibels = np.full((4, 1000), -60.0, np.float32)
    decibels[2, 501] = 0.0
    part = Spectrogram(decibels, 0.01, 24000).cut(5.0, 6.0)
    assert part.decibels.shape == (4, 100) and part.duration_s == 1.0
    assert part.decibels.max() == 0.0


def test_colour_keeps_every_pixel_of_the_spectrogram_with_the_low_end_at_the_bottom() -> None:
    decibels = np.zeros((6, 9), np.float32)
    decibels[0] = 1.0  # lowest frequency, the loudest
    image = colour(decibels, 0.0, 1.0)
    assert image.shape == (6, 9, 3) and image.dtype == np.uint8
    assert image[-1].mean() > image[0].mean()  # bright at the bottom, black at the top


def test_the_quiet_stretch_avoids_annotated_hits() -> None:
    energy = np.zeros(1000, np.float32)
    energy[300:400] = 10.0  # the loudest part, but a hit is annotated there
    energy[700:800] = 5.0
    start = isolated_quiet_stretch(energy, 0.01, [(3.2, 3.6)], 1.0, 0.3)
    assert 6.9 <= start <= 7.1

"""The spectrogram figure of section 4.2.1: what a hit looks like in the sound.

A smash up close (a) and a stretch with no hit (b), the two views the dataset's
own paper (Decorte et al.) uses to show that a hit is a short, sharp stripe and
that not every loud sound is one. Drawn from the audio of a CVSPORTS point with
our own code; nothing is taken from their figure.

The spectrogram is the short-time Fourier transform in decibels with a linear
frequency axis, the picture the reader knows. The network reads a 40-band
log-Mel version of the same sound (padel_ml.audio_dataset).

Two details keep it sharp. The analysis window is short (11 ms), because a hit
lasts a few milliseconds and a longer window smears its stripe into a blur; and
it is zero-padded to give more frequency points without lengthening it. The
image is coloured here and embedded at its own resolution: handed over as raw
numbers, the plotting library shrinks it to the screen's 100 dpi first.

Set at the memoria's text width through the PGF backend, so its text is set by
LaTeX in Latin Modern, like the page around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from padel_ml.thesis.figures import INK, STYLE, TEXT_WIDTH_IN

FloatArray = npt.NDArray[np.float32]
ByteArray = npt.NDArray[np.uint8]

WINDOW = 512
"""Samples per analysis window: 11 ms at 48 kHz."""
N_FFT = 2048
"""FFT length: the window zero-padded to four times its size, for more frequency points."""
HOP = 24
"""Half a millisecond between columns."""
COLORMAP = "magma"
"""Black to yellow, the scale that is read as a spectrogram at a glance."""
TOP_PERCENTILE = 99.0
"""The colour scale tops out at this percentile of the sound, not at its loudest pixel."""
RANGE_DB = 50.0
"""Decibels shown below that top."""
FREQUENCY_TICKS_HZ = (0.0, 12000.0, 24000.0)


@dataclass(frozen=True)
class Spectrogram:
    """Decibels over (frequency, time), starting at time zero."""

    decibels: FloatArray
    """(frequencies, columns), lowest frequency first."""
    column_s: float
    """Seconds between two columns."""
    nyquist_hz: float

    def cut(self, start_s: float, end_s: float) -> Spectrogram:
        """The part between `start_s` and `end_s`, as its own spectrogram from 0."""
        first = max(0, round(start_s / self.column_s))
        last = min(self.decibels.shape[1], round(end_s / self.column_s))
        return Spectrogram(self.decibels[:, first:last], self.column_s, self.nyquist_hz)

    @property
    def duration_s(self) -> float:
        return float(self.decibels.shape[1] * self.column_s)


def compute_spectrogram(
    samples: FloatArray,
    sample_rate: int,
    window: int = WINDOW,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> Spectrogram:
    """Short-time Fourier transform of `samples` in decibels."""
    transform = torch.stft(
        torch.from_numpy(samples),
        n_fft,
        hop,
        win_length=window,
        window=torch.hann_window(window),
        return_complex=True,
    )
    power = transform.abs().pow(2).numpy()
    decibels = (10.0 * np.log10(power + 1e-10)).astype(np.float32)
    return Spectrogram(decibels, hop / sample_rate, sample_rate / 2)


def hertz(value: float) -> str:
    """12000 -> 12\\,000, with the thin space the RAE asks for from five digits."""
    return f"{value:,.0f}".replace(",", r"\,") if value >= 10000 else f"{value:.0f}"


def isolated_quiet_stretch(
    energy: FloatArray,
    column_s: float,
    windows: list[tuple[float, float]],
    length_s: float,
    margin_s: float,
) -> float:
    """Start (s) of the loudest stretch of `length_s` with no annotated hit within `margin_s`."""
    span = round(length_s / column_s)
    sums = np.convolve(energy, np.ones(span), mode="valid")
    best, best_start = -np.inf, 0.0
    for first in range(len(sums)):
        start = first * column_s
        end = start + length_s
        if any(a - margin_s <= end and b + margin_s >= start for a, b in windows):
            continue
        if sums[first] > best:
            best, best_start = float(sums[first]), start
    return best_start


def colour(decibels: FloatArray, low: float, high: float) -> ByteArray:
    """The spectrogram as an RGB image, lowest frequency at the bottom."""
    scaled = np.clip((decibels - low) / (high - low), 0.0, 1.0)
    rgba = mpl.colormaps[COLORMAP](scaled[::-1])
    return np.asarray(rgba[..., :3] * 255, dtype=np.uint8)


def _panel(
    figure: Figure,
    box: tuple[float, float, float, float],
    spectrogram: Spectrogram,
    limits: tuple[float, float],
    ticks_s: list[float],
    caption: str,
) -> Axes:
    left, bottom, width, height = box
    total_w, total_h = figure.get_size_inches()
    axes = figure.add_axes((left / total_w, bottom / total_h, width / total_w, height / total_h))
    axes.imshow(
        colour(spectrogram.decibels, *limits),
        aspect="auto",
        extent=(0, spectrogram.duration_s, 0, spectrogram.nyquist_hz),
        interpolation="none",
    )
    axes.set_xticks(ticks_s)
    axes.set_xticklabels([f"{round(t * 1000)}" for t in ticks_s])
    axes.set_yticks(list(FREQUENCY_TICKS_HZ))
    axes.set_yticklabels([hertz(v) for v in FREQUENCY_TICKS_HZ])
    axes.set_xlabel("Tiempo (ms)", labelpad=2)
    axes.set_ylabel("Frecuencia (Hz)", labelpad=2)
    axes.tick_params(length=2.5, width=0.5, pad=2)
    for edge in axes.spines.values():
        edge.set_linewidth(0.5)
    figure.text(
        (left + width / 2) / total_w,
        (bottom - 0.46) / total_h,
        caption,
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK,
    )
    return axes


def draw_spectrogram_figure(hit: Spectrogram, no_hit: Spectrogram, out: Path) -> None:
    """Write the figure to `out` (PDF): a hit and a stretch with no hit, side by side."""
    both = np.concatenate([hit.decibels.ravel(), no_hit.decibels.ravel()])
    top = float(np.percentile(both, TOP_PERCENTILE))
    limits = (top - RANGE_DB, top)
    wide = TEXT_WIDTH_IN
    side, gap, panel_w, panel_h = 0.62, 0.8, 0.30 * wide, 1.0
    bottom = 0.6
    first = (wide - (2 * panel_w + gap + side)) / 2 + side
    with mpl.rc_context({**STYLE, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK}):
        figure = Figure(figsize=(wide, bottom + panel_h + 0.15))
        for left, spec, ticks, caption in zip(
            (first, first + panel_w + gap),
            (hit, no_hit),
            ([0.0, 0.32, 0.64], [0.0, 0.4, 0.8]),
            ("(a)", "(b)"),
            strict=True,
        ):
            _panel(figure, (left, bottom, panel_w, panel_h), spec, limits, ticks, caption)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

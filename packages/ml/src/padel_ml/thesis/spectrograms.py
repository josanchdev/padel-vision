"""The spectrogram figure of section 4.2.1: what a hit looks like in the sound.

A whole point (a), a smash up close (b) and a stretch with no hit (c), the same
three views the dataset's own paper (Decorte et al.) uses to show that a hit is
a short, sharp stripe and that not every loud sound is one. Drawn from the audio
of a CVSPORTS point with our own code; nothing is taken from their figure.

The spectrogram is the plain short-time Fourier transform in decibels with a
linear frequency axis, the picture the reader knows. The network reads a
40-band log-Mel version of the same sound (padel_ml.audio_dataset).

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

from padel_ml.thesis.figures import MUTED_INK, STYLE
from padel_ml.thesis.problem import TEXT_WIDTH_IN

FloatArray = npt.NDArray[np.float32]

N_FFT = 2048
HOP = 256
"""About 5 ms between columns at 48 kHz: sharp enough for a hit's stripe."""
RANGE_DB = 70.0
"""Decibels shown below the loudest 0.1 % of the spectrogram."""
MAX_COLUMNS = 3000
"""A whole point is shrunk to this many columns, keeping the loudest of each block."""
COLORMAP = "inferno"
"""Black to yellow, the scale that is read as a spectrogram at a glance."""
LABEL_ABOVE_HZ = (0.0, 12000.0, 24000.0)


@dataclass(frozen=True)
class Spectrogram:
    """Decibels over (frequency, time), and where in the sound it starts."""

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

    def shrunk(self, columns: int = MAX_COLUMNS) -> Spectrogram:
        """At most `columns` columns, keeping the loudest of each block."""
        step = int(np.ceil(self.decibels.shape[1] / columns))
        if step <= 1:
            return self
        usable = self.decibels.shape[1] // step * step
        blocks = self.decibels[:, :usable].reshape(self.decibels.shape[0], -1, step)
        return Spectrogram(blocks.max(axis=2), self.column_s * step, self.nyquist_hz)


def compute_spectrogram(samples: FloatArray, sample_rate: int) -> Spectrogram:
    """Short-time Fourier transform of `samples` in decibels."""
    window = torch.hann_window(N_FFT)
    transform = torch.stft(
        torch.from_numpy(samples), N_FFT, HOP, window=window, return_complex=True
    )
    power = transform.abs().pow(2).numpy()
    decibels = (10.0 * np.log10(power + 1e-10)).astype(np.float32)
    return Spectrogram(decibels, HOP / sample_rate, sample_rate / 2)


def clock(seconds: float) -> str:
    """0 -> 0:00, 75 -> 1:15."""
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


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


def _panel(
    figure: Figure,
    box: tuple[float, float, float, float],
    spectrogram: Spectrogram,
    limits: tuple[float, float],
    ticks_s: list[float],
    tick_labels: list[str],
    xlabel: str,
    caption: str,
) -> Axes:
    left, bottom, width, height = box
    total_w, total_h = figure.get_size_inches()
    axes = figure.add_axes((left / total_w, bottom / total_h, width / total_w, height / total_h))
    axes.imshow(
        spectrogram.decibels,
        origin="lower",
        aspect="auto",
        cmap=COLORMAP,
        vmin=limits[0],
        vmax=limits[1],
        extent=(0, spectrogram.duration_s, 0, spectrogram.nyquist_hz),
        interpolation="nearest",
    )
    axes.set_xticks(ticks_s)
    axes.set_xticklabels(tick_labels)
    axes.set_yticks(list(LABEL_ABOVE_HZ))
    axes.set_yticklabels([hertz(v) for v in LABEL_ABOVE_HZ])
    axes.set_xlabel(xlabel, labelpad=2)
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
        color=MUTED_INK,
    )
    return axes


def draw_spectrogram_figure(
    point: Spectrogram | None,
    hit: Spectrogram,
    no_hit: Spectrogram,
    out: Path,
) -> None:
    """Write the figure to `out` (PDF). Without `point`, only the two close-ups."""
    reference = np.concatenate([a.decibels.ravel() for a in (point or hit, hit, no_hit)])
    top = float(np.percentile(reference, 99.9))
    limits = (top - RANGE_DB, top)
    wide = TEXT_WIDTH_IN
    side = 0.62
    small_h = 1.0
    gap = 0.8
    """Between the two close-ups: room for the second one's frequency axis."""
    small_w = (wide - side - 0.1 - gap) / 2 if point is None else 0.38 * wide
    big_h = 1.2
    with mpl.rc_context(STYLE):
        height = small_h + 0.95 + (big_h + 0.95 if point is not None else 0.0)
        figure = Figure(figsize=(wide, height))
        small_bottom = 0.6
        if point is not None:
            shown = point.shrunk()
            seconds = np.arange(0.0, shown.duration_s + 1.0, 15.0)
            seconds = seconds[seconds <= shown.duration_s + 0.5]
            _panel(
                figure,
                (side, small_bottom + small_h + 1.0, wide - side - 0.1, big_h),
                shown,
                limits,
                [float(s) for s in seconds],
                [clock(s) for s in seconds],
                "Tiempo",
                "(a)",
            )
        first = side + ((wide - side - 0.1) - (2 * small_w + gap)) / 2
        lefts = [first, first + small_w + gap]
        for left, spec, ticks, caption in zip(
            lefts,
            (hit, no_hit),
            ([0.0, 0.32, 0.64], [0.0, 0.4, 0.8]),
            ("(b)", "(c)"),
            strict=True,
        ):
            _panel(
                figure,
                (left, small_bottom, small_w, small_h),
                spec,
                limits,
                ticks,
                [f"{round(t * 1000)}" for t in ticks],
                "Tiempo (ms)",
                caption,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

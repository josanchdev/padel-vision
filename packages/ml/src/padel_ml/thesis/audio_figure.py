"""The audio figure of section 4.2.1: how the CRNN finds the hits of a point.

Above, the log-Mel spectrogram the network reads, where a racket hit shows as a
short vertical stripe. Below, the network's hit probability frame by frame, the
detection threshold and the hits the system reports, over the hit windows
annotated in CVSPORTS. The network is the one trained without the point's
tournament (runs/chain_cv), so the figure shows what it does on a tournament it
has not seen.

Set at the memoria's text width through the PGF backend, so its text is set by
LaTeX in Latin Modern, like the page around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from padel_ml.thesis.figures import (
    BASELINE,
    BLUE_RAMP,
    GRID,
    INK,
    MUTED_INK,
    STYLE,
    decimal_axis,
)
from padel_ml.thesis.problem import TEXT_WIDTH_IN

FloatArray = npt.NDArray[np.float32]

HEIGHT_IN = 2.75
ANNOTATED = "#e4e1d8"
"""Neutral warm grey for the annotated hit windows: a reference, not our output."""
DETECTED = "#eb6834"
"""The hits the system reports: the one accent of the figure."""
FREQUENCY_TICKS_KHZ = (1, 2, 5, 10, 20)


@dataclass(frozen=True)
class AudioTrace:
    """What the detector sees and says over one point, frame by frame."""

    times_s: FloatArray
    """(T,) the centre of each spectrogram frame."""
    log_mel: FloatArray
    """(T, n_mels) the network's input, before normalisation."""
    probability: FloatArray
    """(T,) the network's hit probability."""
    detected_s: list[float]
    annotated: list[tuple[float, float]]
    """The hit windows annotated in the dataset, (start, end) in seconds."""
    sample_rate: int
    """Of the audio the spectrogram was computed from."""

    def window(self, start_s: float, end_s: float) -> AudioTrace:
        """The part of the trace between `start_s` and `end_s`."""
        keep = (self.times_s >= start_s) & (self.times_s <= end_s)
        return AudioTrace(
            self.times_s[keep],
            self.log_mel[keep],
            self.probability[keep],
            [t for t in self.detected_s if start_s <= t <= end_s],
            [(a, b) for a, b in self.annotated if b >= start_s and a <= end_s],
            self.sample_rate,
        )


def mel_position(frequency_hz: float, n_mels: int, sample_rate: int) -> float:
    """Where `frequency_hz` falls on the axis of `n_mels` bands (band k at k).

    torchaudio's default (HTK) mel scale, with the bands' centres evenly spaced
    in mel between 0 Hz and the Nyquist frequency."""

    def mel(hz: float) -> float:
        return float(2595.0 * np.log10(1.0 + hz / 700.0))

    return mel(frequency_hz) / (mel(sample_rate / 2) / (n_mels + 1))


def draw_audio_figure(trace: AudioTrace, threshold: float, out: Path) -> None:
    """Write the figure to `out` (PDF)."""
    start, end = float(trace.times_s[0]), float(trace.times_s[-1])
    step = float(np.median(np.diff(trace.times_s)))
    energy = LinearSegmentedColormap.from_list("energia", BLUE_RAMP)
    # The crowd's steady noise stays light, so the hits' broadband stripes stand out
    low, high = np.percentile(trace.log_mel, [30, 99.7])

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, HEIGHT_IN))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=(1.25, 1.0),
            hspace=0.12,
            left=0.075,
            right=0.925,
            top=0.86,
            bottom=0.15,
        )
        spectrum = figure.add_subplot(grid[0])
        curve = figure.add_subplot(grid[1], sharex=spectrum)

        bands = trace.log_mel.shape[1]
        image = spectrum.imshow(
            trace.log_mel.T,
            origin="lower",
            aspect="auto",
            cmap=energy,
            vmin=low,
            vmax=high,
            extent=(start - step / 2, end + step / 2, 0.5, bands + 0.5),
            interpolation="nearest",
        )
        spectrum.set_ylim(0.5, bands + 0.5)
        spectrum.set_yticks(
            [mel_position(khz * 1000, bands, trace.sample_rate) for khz in FREQUENCY_TICKS_KHZ]
        )
        spectrum.set_yticklabels([f"{khz:g}".replace(".", ",") for khz in FREQUENCY_TICKS_KHZ])
        spectrum.set_ylabel("Frecuencia (kHz)")
        # Each detected hit, on top of its stripe, lined up with its peak below
        spectrum.plot(
            trace.detected_s,
            [bands + 2.2] * len(trace.detected_s),
            linestyle="none",
            marker="v",
            markersize=4.5,
            color=DETECTED,
            markeredgecolor="white",
            markeredgewidth=0.6,
            clip_on=False,
        )
        box = spectrum.get_position()
        scale = figure.add_axes((0.94, box.y0, 0.012, box.height))
        bar = figure.colorbar(image, cax=scale, ticks=[low, high])
        bar.ax.set_yticklabels(["baja", "alta"])
        bar.ax.tick_params(length=0, labelsize=7)
        for edge in bar.ax.spines.values():
            edge.set_visible(False)
        bar.set_label("Energía", fontsize=7.5, labelpad=-2)
        spectrum.tick_params(labelbottom=False)
        for side in ("left", "bottom"):
            spectrum.spines[side].set_visible(False)

        for a, b in trace.annotated:
            curve.axvspan(a, b, color=ANNOTATED, linewidth=0, zorder=0)
        curve.axhline(threshold, color=MUTED_INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        curve.plot(trace.times_s, trace.probability, color=INK, linewidth=1.0, zorder=2)
        peaks = [int(np.argmin(np.abs(trace.times_s - t))) for t in trace.detected_s]
        curve.plot(
            trace.times_s[peaks],
            trace.probability[peaks],
            linestyle="none",
            marker="o",
            markersize=4.5,
            color=DETECTED,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        curve.set_ylim(-0.03, 1.05)
        curve.set_yticks([0, threshold, 1])
        curve.yaxis.set_major_formatter(decimal_axis(1))
        curve.set_ylabel("Probabilidad")
        first, last = float(np.floor(start)), float(np.ceil(end))
        curve.set_xlim(first, last)
        curve.set_xticks(np.arange(first, last + 1))
        curve.set_xlabel("Tiempo (s)")
        curve.grid(axis="y", color=GRID, linewidth=0.5)
        curve.spines["bottom"].set_color(BASELINE)

        figure.legend(
            handles=[
                Patch(facecolor=ANNOTATED, edgecolor="none"),
                Line2D([], [], color=INK, linewidth=1.0),
                Line2D([], [], color=MUTED_INK, linewidth=0.8, linestyle=(0, (3, 2))),
                Line2D([], [], linestyle="none", marker="o", markersize=4.5, color=DETECTED),
            ],
            labels=["Golpe anotado", "Probabilidad de golpe", "Umbral", "Golpe detectado"],
            loc="upper center",
            ncol=4,
            frameon=False,
            handlelength=1.4,
            columnspacing=1.6,
            bbox_to_anchor=(0.53, 1.0),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

"""Write the spectrogram figure of section 4.2.1: memoria/figuras/espectrogramas.pdf.

Two panels from the audio of one CVSPORTS point: a smash up close and a stretch
with no hit, side by side.

    uv run python scripts/build_spectrogram_figure.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from padel_ml.audio_dataset import SAMPLE_RATE, decode_audio
from padel_ml.thesis.spectrograms import (
    Spectrogram,
    compute_spectrogram,
    draw_spectrogram_figure,
    isolated_quiet_stretch,
)

from padel_cv.paths import CVSPORTS_HITS, CVSPORTS_RALLIES, REPO

OUT = REPO / "memoria" / "figuras" / "espectrogramas.pdf"

RALLY = "20230528_VIGO_11"
"""A point of 61 s, the one with the most hits of the Vigo group."""
SMASH_S = 20.16
"""A smash with no other hit within 1.6 s, so that its stripe stands alone. The annotated
instant is the middle of a window, so the stripe itself is found next to it."""
HIT_BEFORE_S, HIT_LENGTH_S = 0.32, 0.68
EDGE_S = 3.0
"""The no-hit stretch is looked for away from the point's start and end, where the
serve preparation and the crowd after the last hit would be picked."""
NO_HIT_LENGTH_S = 0.8
NO_HIT_MARGIN_S = 0.3
"""The stretch with no hit is the loudest 0.8 s with no annotated hit within 0.3 s."""
SEARCH_HOP = 256
"""A coarser spectrogram is enough to find the stripe and the stretch."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    samples = decode_audio(CVSPORTS_RALLIES / f"{RALLY}.mp4")
    overview = compute_spectrogram(samples, SAMPLE_RATE, 1024, 1024, SEARCH_HOP)
    with CVSPORTS_HITS.open() as handle:
        windows = sorted(
            (float(r["start"]), float(r["end"]))
            for r in csv.DictReader(handle)
            if r["filename"] == f"{RALLY}.mp4"
        )
    bins = overview.decibels.shape[0]

    def band(low_hz: float, high_hz: float) -> np.ndarray:
        rows = slice(
            round(low_hz / overview.nyquist_hz * bins), round(high_hz / overview.nyquist_hz * bins)
        )
        return overview.decibels[rows].mean(axis=0)

    # The hit's stripe: the sharpest rise in the 4-12 kHz band within 0.3 s of the annotation
    rise = np.diff(band(4000, 12000))
    first = round((SMASH_S - 0.3) / overview.column_s)
    around = rise[first : first + round(0.6 / overview.column_s)]
    stripe_s = (first + int(np.argmax(around))) * overview.column_s
    # The quiet stretch: the loudest one between 100 Hz and 4 kHz, where voices live
    loudness = band(100, 4000).astype(np.float32)
    loudness[: round(EDGE_S / overview.column_s)] = -np.inf
    loudness[round((overview.duration_s - 4.0) / overview.column_s) :] = -np.inf
    start = isolated_quiet_stretch(
        loudness, overview.column_s, windows, NO_HIT_LENGTH_S, NO_HIT_MARGIN_S
    )

    def close_up(start_s: float, length_s: float) -> Spectrogram:
        part = samples[round(start_s * SAMPLE_RATE) : round((start_s + length_s) * SAMPLE_RATE)]
        return compute_spectrogram(part, SAMPLE_RATE)

    print(f"golpe: {stripe_s:.2f} s · tramo sin golpe: {start:.2f}-{start + NO_HIT_LENGTH_S:.2f} s")
    draw_spectrogram_figure(
        close_up(stripe_s - HIT_BEFORE_S, HIT_LENGTH_S),
        close_up(start, NO_HIT_LENGTH_S),
        args.out,
    )
    print(args.out)


if __name__ == "__main__":
    main()

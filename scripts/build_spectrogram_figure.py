"""Write the spectrogram figure of section 4.2.1: memoria/figuras/espectrogramas.pdf.

Three panels from the audio of one CVSPORTS point: the whole point, a smash up
close and a stretch with no hit. `--compact` writes only the two close-ups, side
by side, to memoria/figuras/espectrogramas_zoom.pdf.

    uv run python scripts/build_spectrogram_figure.py [--compact]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from padel_ml.audio_dataset import SAMPLE_RATE, decode_audio
from padel_ml.thesis.spectrograms import (
    compute_spectrogram,
    draw_spectrogram_figure,
    isolated_quiet_stretch,
)

from padel_cv.paths import CVSPORTS_HITS, CVSPORTS_RALLIES, REPO

OUT = REPO / "memoria" / "figuras" / "espectrogramas.pdf"
OUT_COMPACT = REPO / "memoria" / "figuras" / "espectrogramas_zoom.pdf"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    samples = decode_audio(CVSPORTS_RALLIES / f"{RALLY}.mp4")
    spectrogram = compute_spectrogram(samples, SAMPLE_RATE)
    with CVSPORTS_HITS.open() as handle:
        windows = sorted(
            (float(r["start"]), float(r["end"]))
            for r in csv.DictReader(handle)
            if r["filename"] == f"{RALLY}.mp4"
        )
    # Loudness of each column between 100 Hz and 4 kHz, where voices and hits live
    bins = spectrogram.decibels.shape[0]
    low, high = (
        round(100 / spectrogram.nyquist_hz * bins),
        round(4000 / spectrogram.nyquist_hz * bins),
    )
    energy = spectrogram.decibels[low:high].mean(axis=0).astype(np.float32)
    # The hit's stripe: the sharpest rise in the 4-12 kHz band within 0.3 s of the annotation
    band = spectrogram.decibels[
        round(4000 / spectrogram.nyquist_hz * bins) : round(12000 / spectrogram.nyquist_hz * bins)
    ].mean(axis=0)
    around = slice(
        round((SMASH_S - 0.3) / spectrogram.column_s), round((SMASH_S + 0.3) / spectrogram.column_s)
    )
    stripe_s = (around.start + int(np.argmax(np.diff(band[around])))) * spectrogram.column_s
    hit = spectrogram.cut(stripe_s - HIT_BEFORE_S, stripe_s - HIT_BEFORE_S + HIT_LENGTH_S)
    quiet = energy.copy()
    quiet[: round(EDGE_S / spectrogram.column_s)] = -np.inf
    quiet[round((spectrogram.duration_s - 4.0) / spectrogram.column_s) :] = -np.inf
    start = isolated_quiet_stretch(
        quiet, spectrogram.column_s, windows, NO_HIT_LENGTH_S, NO_HIT_MARGIN_S
    )
    no_hit = spectrogram.cut(start, start + NO_HIT_LENGTH_S)
    print(f"golpe: {stripe_s:.2f} s · tramo sin golpe: {start:.2f}-{start + NO_HIT_LENGTH_S:.2f} s")
    out = args.out or (OUT_COMPACT if args.compact else OUT)
    draw_spectrogram_figure(None if args.compact else spectrogram, hit, no_hit, out)
    print(out)


if __name__ == "__main__":
    main()

"""Write the audio figure of section 4.2.1: memoria/figuras/audio.pdf.

Runs the audio detector trained without the point's tournament (the full-system
evaluation's model in runs/chain_cv) over one CVSPORTS point. Needs the video and
a few seconds of CPU or GPU.

    uv run python scripts/build_audio_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from padel_ml.audio_dataset import build_rally, frame_time
from padel_ml.audio_train import DEFAULT_THRESHOLD, hit_probabilities, peaks_from_frames
from padel_ml.thesis.audio_figure import AudioTrace, draw_audio_figure

from padel_cv.cvsports import load_hits_csv, tournament_of
from padel_cv.paths import CVSPORTS_HITS, CVSPORTS_RALLIES, REPO, RUNS

OUT = REPO / "memoria" / "figuras" / "audio.pdf"

RALLY = "20230528_VIGO_14"
"""The same point as the problem figure."""
START_S, END_S = 0.0, 9.0
"""Its first six hits: the serve and five more."""
MIN_GAP_FRAMES = 8
"""Frames between two reported hits, as the system uses them (detect_hits_in_audio)."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    video = CVSPORTS_RALLIES / f"{RALLY}.mp4"
    checkpoint = RUNS / "chain_cv" / tournament_of(RALLY) / "audio.pt"
    probability = hit_probabilities(video, checkpoint)
    log_mel = build_rally(video, []).features
    times = np.array([frame_time(i) for i in range(len(probability))], dtype=np.float32)
    peaks = peaks_from_frames(probability, DEFAULT_THRESHOLD, MIN_GAP_FRAMES)
    trace = AudioTrace(
        times,
        log_mel[: len(probability)],
        probability,
        [frame_time(p) for p in peaks],
        load_hits_csv(CVSPORTS_HITS)[f"{RALLY}.mp4"],
    )
    draw_audio_figure(trace.window(START_S, END_S), DEFAULT_THRESHOLD, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

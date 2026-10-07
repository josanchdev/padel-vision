"""Write the problem figure of chapter 2: memoria/figuras/problema.pdf.

Runs the delivered system (the trained models under runs/) over one CVSPORTS
point, so it needs the video and takes about half a minute on a GPU.

    uv run python scripts/build_problem_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.rally_render import annotated_frame
from padel_ml.thesis.problem import draw_problem_figure

from padel_cv.court_registry import court_file_for
from padel_cv.paths import CVSPORTS_RALLIES, REPO, RUNS
from padel_cv.video_io import read_frame

OUT = REPO / "memoria" / "figuras" / "problema.pdf"

RALLY = "20230528_VIGO_14"
SHOWN_S = 4.52
"""J3's backhand, near the camera, with the ball trail arriving at the racket.
The system gets it right (instant, player and type), as it does with the hits
around it."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    court = court_file_for(RALLY)
    if court is None:
        raise SystemExit(f"{RALLY}: sin pista marcada")
    device = args.device if torch.cuda.is_available() else "cpu"
    video = CVSPORTS_RALLIES / f"{RALLY}.mp4"
    analysis = analyze_rally(video, RallyModels(ModelPaths.under(RUNS), device=device), court)
    shot = min(analysis.shots, key=lambda shot: abs(shot.timestamp_s - SHOWN_S))

    raw = cv2.cvtColor(read_frame(video, shot.frame_index), cv2.COLOR_BGR2RGB)
    processed = cv2.cvtColor(annotated_frame(analysis, shot.frame_index), cv2.COLOR_BGR2RGB)
    draw_problem_figure(raw, processed, args.out)
    print(f"{args.out}: J{shot.player_id} {shot.shot_type} en el fotograma {shot.frame_index}")


if __name__ == "__main__":
    main()

"""Write the ball heatmap figure of section 4.2.3: memoria/figuras/mapa-calor.pdf.

Runs the system's trained ball network (runs/ball_full) over three consecutive
frames of one CVSPORTS point and draws the heatmap it returns for the last one.

    uv run python scripts/build_heatmap_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.ball_infer import build_ball_model
from padel_ml.thesis.heatmap_figure import THRESHOLD, ball_heatmap, draw_heatmap_figure, peak_cell

from padel_cv.paths import CVSPORTS_RALLIES, REPO, RUNS
from padel_cv.video_io import read_frame

OUT = REPO / "memoria" / "figuras" / "mapa-calor.pdf"

RALLY = "20230927_MADRID_01"
"""The Madrid point of the player figure and the web viewer."""
SHOWN_FRAME = 283
"""The ball alone on the near half of the court, sharp and away from every
player, detected with a peak of 0.97."""
CHECKPOINT = RUNS / "ball_full" / "tracknetv3.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_ball_model(stored.get("model_name", "tracknetv2"))
    model.load_state_dict(stored["state_dict"])
    model.eval()

    video = CVSPORTS_RALLIES / f"{RALLY}.mp4"
    frames = [read_frame(video, SHOWN_FRAME - offset) for offset in (2, 1, 0)]
    heatmap = ball_heatmap(model, frames)
    column, row = peak_cell(heatmap)
    peak = float(heatmap[row, column])
    if peak < THRESHOLD:
        raise SystemExit(f"fotograma {SHOWN_FRAME}: máximo {peak:.2f}, no hay pelota")
    draw_heatmap_figure(frames[-1], heatmap, args.out)
    cells = int((heatmap >= THRESHOLD).sum())
    print(f"máximo {peak:.2f} en la celda ({column}, {row}); {cells} celdas >= {THRESHOLD}")
    print(args.out)


if __name__ == "__main__":
    main()

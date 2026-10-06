"""Write the screenshots of the dissertation that come from running the system.

Unlike build_thesis_assets.py, these need the trained models (runs/) and the
CVSPORTS videos, and a GPU to take about a minute per rally. Every image is
drawn by the system's own code on a frame of the original video, never taken
from the screen or from the rendered video (memoria/GUIA.md, section 5).

    uv run python scripts/build_thesis_captures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.rally_render import annotated_frame

from padel_cv.court_registry import court_file_for
from padel_cv.paths import CVSPORTS_RALLIES, REPO, RUNS

OUT = REPO / "memoria" / "figuras" / "capturas"

ANNOTATED_RALLY = "20230528_VIGO_11"
ANNOTATED_SHOT = 20
"""A smash by J4, the near right player, with the ball trail arriving at him."""
ANNOTATED_OFFSET = 1
"""Frames after the hit: the verdict is on screen and the trail still shows the
ball coming in."""
JPEG_QUALITY = 95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    models = RallyModels(ModelPaths.under(RUNS), device=device)
    video = CVSPORTS_RALLIES / f"{ANNOTATED_RALLY}.mp4"
    court = court_file_for(ANNOTATED_RALLY)
    if court is None:
        raise SystemExit(f"{ANNOTATED_RALLY}: sin pista marcada")
    analysis = analyze_rally(video, models, court)
    shot = analysis.shots[ANNOTATED_SHOT]
    image = annotated_frame(analysis, shot.frame_index + ANNOTATED_OFFSET)
    target = out / "frame-anotado.jpg"
    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"  {target.relative_to(REPO)}  ({ANNOTATED_RALLY}, golpe {ANNOTATED_SHOT}: "
          f"J{shot.player_id} {shot.shot_type} {shot.confidence:.2f})")  # fmt: skip


if __name__ == "__main__":
    main()

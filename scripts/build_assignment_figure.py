"""Write the hit assignment figure of section 4.2.4.

memoria/figuras/asignacion-fotograma.pdf (the frame) and memoria/figuras/asignacion.tex
(the frame beside its table). Runs the system's per-frame pass (pose, identity and
ball) over one VIGO point and reads one frame of it, so it needs the video and
takes about a minute on a GPU.

    uv run python scripts/build_assignment_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.ball_infer import BallDetector
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.rally_analysis import POSE_CONFIDENCE, track_rally
from padel_ml.thesis.assignment_figure import draw_frame, figure_tex, measure

from padel_cv.court_registry import court_file_for
from padel_cv.paths import CVSPORTS_RALLIES, REPO, RUNS
from padel_cv.pose import PoseDetector
from padel_cv.video_io import read_frame

OUT_DIR = REPO / "memoria" / "figuras"
FRAME_PDF = "asignacion-fotograma"

RALLY = "20230528_VIGO_02"
SHOWN_FRAME = 167
"""A smash by J4 (near side), at the annotated instant of the hit. In pixels the
ball is nearer J2 (far side, small in the image); in heights, nearer J4."""
HITTER = 4
"""Who hit, as the dataset's per-hit annotation says (hit_assignments.xlsx)."""
CROP = (560, 240, 1300, 820)
"""x0, y0, x1, y1 in frame pixels: the four players and the ball."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video = CVSPORTS_RALLIES / f"{RALLY}.mp4"
    tracks = track_rally(
        video,
        PoseDetector(confidence=POSE_CONFIDENCE, device=device),
        BallDetector(RUNS / "ball_full" / "tracknetv3.pt", device=device),
        court_file_for(RALLY),
    )
    ball_track = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(tracks.raw_ball)}
    ball = ball_track[SHOWN_FRAME]
    poses = [p for p in tracks.players[SHOWN_FRAME] if p.player_id is not None]
    measures = measure(poses, ball)
    if min(measures, key=lambda m: m.heights).player != HITTER:
        raise SystemExit(f"fotograma {SHOWN_FRAME}: en alturas no gana J{HITTER}")

    draw_frame(
        read_frame(video, SHOWN_FRAME),
        poses,
        ball,
        CROP,
        args.out_dir / f"{FRAME_PDF}.pdf",
        labelled=HITTER,
    )
    (args.out_dir / "asignacion.tex").write_text(figure_tex(f"figuras/{FRAME_PDF}", measures))
    for m in measures:
        print(f"  J{m.player}: {m.pixels:6.1f} px / {m.height:5.1f} px = {m.heights:.2f}")
    print(args.out_dir / "asignacion.tex")


if __name__ == "__main__":
    main()

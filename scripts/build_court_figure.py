"""Write the player figure of section 4.2.2: memoria/figuras/jugadores.pdf.

Runs the delivered system (the trained models under runs/) over one CVSPORTS
point and draws one instant of it: the court mask with the four identified
skeletons, and their positions in metres with the distances between partners
and to the net. Needs the video and takes about a minute on a GPU.

    uv run python scripts/build_court_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.thesis.court_figure import Player, draw_court_figure, net_distance, partner_distances

from padel_cv.court_registry import court_file_for, load_corners
from padel_cv.paths import CVSPORTS_RALLIES, REPO, RUNS
from padel_cv.player_identity import court_mask_polygon
from padel_cv.video_io import read_frame

OUT = REPO / "memoria" / "figuras" / "jugadores.pdf"

RALLY = "20230927_MADRID_01"
"""The Madrid point the web viewer shows."""
SHOWN_FRAME = 200
"""8.0 s into the point, between two hits: the four players waiting in position,
none of them mid-stroke."""


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

    players = [
        Player(pose.player_id, pose.keypoints, pose.bbox_xyxy, pose.court_position_m)
        for pose in sorted(analysis.players[SHOWN_FRAME], key=lambda pose: pose.player_id or 0)
        if pose.player_id is not None and pose.court_position_m is not None
    ]
    if len(players) != 4:
        raise SystemExit(f"fotograma {SHOWN_FRAME}: {len(players)} jugadores, se esperaban 4")
    polygon = court_mask_polygon(load_corners(court))
    draw_court_figure(read_frame(video, SHOWN_FRAME), polygon, players, args.out)

    for (a, b), distance in partner_distances(players).items():
        print(f"  J{a}-J{b}: {distance:.2f} m")
    for player in players:
        print(f"  J{player.number} a la red: {net_distance(player):.2f} m")
    print(args.out)


if __name__ == "__main__":
    main()

"""Render the whole pipeline over one rally: when, who, and what kind of shot.

A thin wrapper over `padel_ml.rally_analysis` — the same code the external
evaluation and the web worker run — plus the overlay from `padel_ml.rally_render`.

Note on honesty: with a CVSPORTS rally the classifier has seen these hits during
training, so that is a "watch it work" demo, not a measurement. The measurement
is the cross-tournament evaluation in docs/metrics (accuracy 81.84%, macro-F1
0.847); on footage from outside CVSPORTS it is the external evaluation.

    uv run python scripts/demo_full_pipeline.py RALLY.mp4 -o out.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.rally_render import SPANISH, render_rally

from padel_cv.court_registry import court_file_for

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rally", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Reject below this confidence (0 = never reject)",
    )
    parser.add_argument(
        "--court", type=Path, help="Court JSON (default: resolved from the video name)"
    )
    parser.add_argument("--data", type=Path, help="Also write the web's JSON here")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    court = args.court or court_file_for(args.rally.stem)
    if court is None:
        # Loud on purpose: without the mask the crowd counts as players, and a
        # spectator can win the vote for a hit. Easy to miss in a silent run.
        print(
            f"[aviso] sin pista marcada para '{args.rally.stem}' — el publico NO se filtra.\n"
            f"        marcala con: uv run padel-cv annotate-court {args.rally} "
            f"-o data/datasets/courts/{args.rally.stem}.json"
        )

    device = args.device if torch.cuda.is_available() else "cpu"
    models = RallyModels(ModelPaths.under(REPO / "runs"), device=device)
    analysis = analyze_rally(args.rally, models, court, min_confidence=args.threshold)

    coverage = 100 * len(analysis.ball) / max(analysis.n_frames, 1)
    print(f"[audio] {len(analysis.shots)} golpes detectados")
    print(f"[cv] {analysis.n_frames} frames · pelota en {len(analysis.ball)} ({coverage:.0f}%)")
    for shot in analysis.shots:
        name = SPANISH.get(shot.shot_type or "", "SIN CLASIFICAR")
        print(
            f"  golpe f={shot.frame_index:5d}  J{shot.player_id}  {name:14s} "
            f"({shot.confidence:.2f})"
        )

    if args.data is not None:
        analysis.to_match_data().write_json(args.data)
    render_rally(analysis, args.out)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()

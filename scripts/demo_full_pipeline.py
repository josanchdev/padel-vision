"""Render the whole pipeline over one rally: when, who, and what kind of shot.

A thin wrapper over `padel_ml.rally_analysis` — the same code the web export
runs — plus the overlay from `padel_ml.rally_render`.

Note on honesty: the production models were trained on all of CVSPORTS, so on a
CVSPORTS rally this is a "watch it work" demo, not a measurement. The
measurements are the cross-tournament evaluations in docs/metrics
(`shot_type_classifier.json`, `chain_cv.json`).

    uv run python scripts/demo_full_pipeline.py RALLY.mp4 -o out.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.rally_render import SPANISH, render_rally

from padel_cv.court_annotator import court_or_mark
from padel_cv.paths import RUNS


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

    court = args.court or court_or_mark(args.rally)
    if court is None:
        # Without the mask the crowd counts as players, and a spectator can win
        # the vote for a hit: better no demo than a silently wrong one.
        parser.error(f"{args.rally.stem}: sin pista marcada (marcado cancelado)")

    device = args.device if torch.cuda.is_available() else "cpu"
    models = RallyModels(ModelPaths.under(RUNS), device=device)
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

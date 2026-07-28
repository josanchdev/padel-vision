"""CLI: cross-court generalization report for the ball detector (Phase 1d).

    padel-ball-generalization --model CKPT --court NAME=VIDEO [--court ...] \
        [-o runs/ball_gen] [--max-frames N] [--videos]

Runs the detector on each court and reports detection rate + mean confidence,
writing a table, JSON, a bar chart, and (with --videos) a verification mp4 each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padel_ml.ball_generalization import run_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="padel-ball-generalization", description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="TrackNet checkpoint")
    parser.add_argument(
        "--court",
        action="append",
        required=True,
        metavar="NAME=VIDEO",
        help="A court to evaluate (repeatable), e.g. wpt=data/raw/2022_BCN_FinalM_1.mp4",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("runs/ball_gen"))
    parser.add_argument("--conf", type=float, default=0.5, help="Min confidence")
    parser.add_argument(
        "--max-frames", type=int, default=3000, help="Frames per court (fair comparison)"
    )
    parser.add_argument("--videos", action="store_true", help="Also write verification mp4s")
    args = parser.parse_args()

    courts: dict[str, Path] = {}
    for spec in args.court:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--court must be NAME=VIDEO, got {spec!r}")
        courts[name] = Path(path)

    run_report(
        courts,
        args.model,
        args.out,
        min_confidence=args.conf,
        max_frames=args.max_frames,
        write_videos=args.videos,
    )


if __name__ == "__main__":
    main()

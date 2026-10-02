"""Process rallies and export them for the web viewer (ADR-0017).

Runs the whole system on each video (`analyze_rally`, the same code as the demo)
and writes what the viewer reads: see `padel_ml.web_export`. The viewer has no
backend, so this is how a point gets into it. A video whose tournament has no
court marked yet opens the marking tool first (ADR-0018).

    uv run python scripts/export_points.py RALLY.mp4 [MORE.mp4 ...]
    uv run python scripts/export_points.py VIDEO --court COURT.json --title "Final Menorca"

Points already exported are skipped unless --force, so a crash (WSL) resumes
where it stopped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally
from padel_ml.web_export import describe, export_point, write_index

from padel_cv.court_annotator import court_or_mark
from padel_cv.paths import RUNS, WEB_POINTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--court", type=Path, help="Court JSON (only with a single video)")
    parser.add_argument("--title", help="Name shown in the viewer (only with a single video)")
    parser.add_argument("--out", type=Path, default=WEB_POINTS)
    parser.add_argument("--force", action="store_true", help="Re-export points already there")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.videos) > 1 and (args.court or args.title):
        parser.error("--court and --title apply to a single video")

    device = args.device if torch.cuda.is_available() else "cpu"
    models = RallyModels(ModelPaths.under(RUNS), device=device)
    for video in args.videos:
        if (args.out / video.stem / "video.mp4").exists() and not args.force:
            print(f"[{video.stem}] ya exportado")
            continue
        court = args.court or court_or_mark(video)
        if court is None:
            # The viewer's court panel and heatmaps need positions in metres.
            parser.error(f"{video.stem}: sin pista marcada (marcado cancelado)")
        started = time.perf_counter()
        analysis = analyze_rally(video, models, court)
        folder = export_point(analysis, args.out, describe(video.stem, args.title))
        print(
            f"[{video.stem}] {len(analysis.shots)} golpes -> {folder}"
            f"  ({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
    print(f"[index] {write_index(args.out)}")


if __name__ == "__main__":
    main()

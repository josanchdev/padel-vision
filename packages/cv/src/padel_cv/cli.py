"""Command-line entry point: the labelling tools and the ball-training cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _dataset_progress(hits_csv: Path, out_dir: Path, skip: str = "") -> tuple[int, int]:
    """(labelled so far, total hits) across the dataset, excluding `skip`.

    Seeing only "3/16" hides how the session fits into the 2,377-hit whole.
    """
    import csv as _csv

    from padel_cv.shot_type_annotator import load_marks

    total = sum(1 for _ in _csv.DictReader(hits_csv.open()))
    done = 0
    if out_dir.is_dir():
        for csv_file in out_dir.glob("*.csv"):
            if csv_file.stem == skip:
                continue
            done += sum(1 for m in load_marks(csv_file).values() if m.shot_type)
    return done, total


def _next_unlabelled_rally(rally_dir: Path, hits_csv: Path, out_dir: Path) -> Path | None:
    """First rally under `rally_dir` that still has hits without a type.

    Labelling 99 rallies one command at a time is friction; this lets the same
    command be run over and over to work through the dataset in order.
    """
    from padel_cv.shot_type_annotator import load_hit_frames, load_marks

    for video in sorted(p for p in rally_dir.glob("*.mp4")):
        frames = load_hit_frames(hits_csv, video.name, fps=25.0)
        if not frames:
            continue
        marks = load_marks(out_dir / f"{video.stem}.csv") if out_dir.is_dir() else {}
        if any(f not in marks or marks[f].shot_type is None for f in frames):
            return video
    return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="padel-cv", description="Padel Vision tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ball = subparsers.add_parser(
        "extract-ball-frames",
        help="Cache downscaled frames + ball centres for TrackNet (resumable shards)",
    )
    ball.add_argument("video", type=Path, help="Match video path")
    ball.add_argument("--ball", type=Path, required=True, help="PadelTracker100 *_ball.json")
    ball.add_argument("-o", "--cache-dir", type=Path, required=True, help="Shard output dir")
    ball.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    types = subparsers.add_parser(
        "annotate-types",
        help="Label the TYPE of already-located hits (ADR-0016): jump, look, tap a key",
    )
    types.add_argument("video", type=Path, help="Rally video")
    types.add_argument("-o", "--out", type=Path, required=True, help="Output types CSV")
    types.add_argument(
        "--hits",
        type=Path,
        default=None,
        help="hits.csv with the hit instants (CVSPORTS GT). Defaults to the dataset's.",
    )
    types.add_argument(
        "--next",
        action="store_true",
        help="Treat `video` as a directory: open the next rally with unlabelled hits",
    )

    court = subparsers.add_parser(
        "annotate-court",
        help="Mark the court by hand for a tournament (6 clicks, ADR-0015 D2)",
    )
    court.add_argument("video", type=Path, help="One rally of the tournament (camera is fixed)")
    court.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output court JSON (default: the tournament's file in data/datasets/courts/)",
    )
    court.add_argument("--frame", type=int, default=30, help="Frame to show")

    args = parser.parse_args()
    if args.command == "extract-ball-frames":
        from padel_cv.ball_cache import extract_ball_frames_to_cache

        extract_ball_frames_to_cache(
            args.video, args.ball, args.cache_dir, max_frames=args.max_frames
        )
    elif args.command == "annotate-court":
        import json as _json

        from padel_cv.court_annotator import annotate_court
        from padel_cv.court_registry import COURTS_DIR, tournament_of

        out = args.out or COURTS_DIR / f"{tournament_of(args.video.stem)}.json"
        homography = annotate_court(args.video, out, frame_index=args.frame)
        if homography is None:
            print("cancelado")
        else:
            error = _json.loads(out.read_text())["reprojection_error_m"]
            print(f"guardado -> {out}  (error de reproyeccion {error} m)")
    elif args.command == "annotate-types":
        import collections

        import cv2

        from padel_cv.shot_type_annotator import annotate_types, load_hit_frames, load_marks

        hits_csv = args.hits or (
            Path("data/raw/padel_audio_dataset/CVSPORTS_Padel/metadata/hits.csv")
        )
        video = args.video
        if args.next:  # pick up where the last session stopped
            video = _next_unlabelled_rally(args.video, hits_csv, args.out)
            if video is None:
                print("Todos los rallies estan etiquetados.")
                return 0
            print(f"Siguiente rally: {video.name}")
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        frames = load_hit_frames(hits_csv, video.name, fps)
        if not frames:
            parser.error(f"No hits for {video.name} in {hits_csv}")
        out_csv = args.out / f"{video.stem}.csv" if args.out.is_dir() else args.out
        context = _dataset_progress(hits_csv, args.out, skip=video.stem)
        print(f"{len(frames)} golpes localizados en {video.name} — solo falta el tipo")
        print(f"progreso del dataset: {context[0]} / {context[1]} golpes etiquetados")
        annotate_types(video, frames, out_csv, total_context=context)
        done = load_marks(out_csv)
        counts = collections.Counter(m.shot_type for m in done.values() if m.shot_type)
        print(f"\n{sum(counts.values())}/{len(frames)} etiquetados -> {out_csv}")
        print("por tipo:", dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

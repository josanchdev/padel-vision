"""Command-line entry point: process a video and write the annotated result."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2

from padel_cv.pipeline import Pipeline, PipelineStage
from padel_cv.stages import (
    CourtDetectionStage,
    GroundTruthCourtStage,
    PlayerIdentityStage,
    PlayerPoseStage,
)
from padel_cv.stages.pose import DEFAULT_TRACKER
from padel_cv.visualize import draw_poses, overlay_minimap


def process_video(
    input_path: Path,
    output_path: Path,
    model_name: str,
    confidence: float,
    image_size: int,
    max_frames: int | None,
    tracker: str | None = "bytetrack.yaml",
    homography_json: Path | None = None,
    court_model: str | None = None,
) -> int:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    stages: list[PipelineStage] = [
        PlayerPoseStage(
            model_name=model_name,
            confidence=confidence,
            image_size=image_size,
            tracker=tracker,
        )
    ]
    court_stage: PipelineStage | None = None
    if court_model is not None:
        court_stage = CourtDetectionStage(court_model)
    elif homography_json is not None:
        court_stage = GroundTruthCourtStage(homography_json)
    if court_stage is not None:
        stages.append(court_stage)
        stages.append(PlayerIdentityStage())
    pipeline = Pipeline(stages)
    track_ids_seen: set[int] = set()
    start = time.perf_counter()
    frames_written = 0
    try:
        for frame in pipeline.run(str(input_path)):
            track_ids_seen.update(p.track_id for p in frame.poses if p.track_id is not None)
            writer.write(overlay_minimap(draw_poses(frame), frame))
            frames_written += 1
            if max_frames is not None and frames_written >= max_frames:
                break
            if frames_written % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"  {frames_written} frames ({frames_written / elapsed:.1f} fps)")
    finally:
        writer.release()
    elapsed = time.perf_counter() - start
    print(f"Wrote {frames_written} frames to {output_path} in {elapsed:.1f}s")
    if track_ids_seen:
        print(f"Track IDs seen: {sorted(track_ids_seen)}")
    return frames_written


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def sample_frames(input_dir: Path, output_dir: Path, per_video: int, seed: int) -> int:
    """Extract random frames from every video under input_dir for annotation.

    Diversity beats volume for the court-keypoint dataset (ADR-0005): a few
    frames from many different courts/views generalize better than many frames
    from one video.
    """
    videos = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    saved = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            capture.release()
            print(f"  skipping unreadable video: {video.name}")
            continue
        # Avoid intros/outros: sample from the middle 90% of the video.
        low, high = int(total * 0.05), max(int(total * 0.95), 1)
        indices = sorted(rng.sample(range(low, high), min(per_video, high - low)))
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                continue
            out_path = output_dir / f"{video.stem}_{index:06d}.png"
            cv2.imwrite(str(out_path), image)
            saved += 1
        capture.release()
    print(f"Saved {saved} frames from {len(videos)} videos to {output_dir}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(prog="padel-cv", description="Padel Vision CV pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Run the pipeline on a video")
    process.add_argument("input", type=Path, help="Input video path")
    process.add_argument("-o", "--output", type=Path, required=True, help="Annotated output path")
    process.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose model")
    process.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    process.add_argument(
        "--imgsz", type=int, default=1920, help="Inference resolution (long side, px)"
    )
    process.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER,
        help="Tracker config YAML (default: padel-tuned ByteTrack) or 'none' to disable",
    )
    process.add_argument(
        "--homography",
        type=Path,
        default=None,
        help="PadelTracker100 homography JSON (dev/eval only): enables minimap + court filter",
    )
    process.add_argument(
        "--court-model",
        default=None,
        help="Trained court-keypoint model: automatic homography on any video (overrides GT)",
    )

    build = subparsers.add_parser(
        "build-court-dataset",
        help="Auto-label court keypoints from PadelTracker100 GT homographies",
    )
    build.add_argument("video", type=Path, help="Match video path")
    build.add_argument("--homography", type=Path, required=True, help="GT homography JSON")
    build.add_argument("-o", "--output", type=Path, required=True, help="Dataset root dir")
    build.add_argument("--split", default="train", help="Dataset split (train/val)")
    build.add_argument("--every", type=int, default=150, help="Sample every N frames")

    sample = subparsers.add_parser(
        "sample-frames", help="Extract random frames from videos for annotation"
    )
    sample.add_argument("input_dir", type=Path, help="Directory containing videos")
    sample.add_argument("-o", "--output", type=Path, required=True, help="Output frames dir")
    sample.add_argument("--per-video", type=int, default=5, help="Frames sampled per video")
    sample.add_argument("--seed", type=int, default=42, help="Random seed (reproducible batches)")
    process.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    args = parser.parse_args()
    if args.command == "process":
        tracker = None if args.tracker == "none" else args.tracker
        process_video(
            args.input,
            args.output,
            args.model,
            args.conf,
            args.imgsz,
            args.max_frames,
            tracker,
            args.homography,
            args.court_model,
        )
    elif args.command == "build-court-dataset":
        from padel_cv.datasets import build_court_dataset

        build_court_dataset(args.video, args.homography, args.output, args.split, args.every)
    elif args.command == "sample-frames":
        sample_frames(args.input_dir, args.output, args.per_video, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

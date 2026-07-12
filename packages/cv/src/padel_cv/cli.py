"""Command-line entry point: process a video and write the annotated result."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from padel_cv.pipeline import Pipeline
from padel_cv.stages import PlayerPoseStage
from padel_cv.visualize import draw_poses


def process_video(
    input_path: Path, output_path: Path, model_name: str, confidence: float, max_frames: int | None
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

    pipeline = Pipeline([PlayerPoseStage(model_name=model_name, confidence=confidence)])
    start = time.perf_counter()
    frames_written = 0
    try:
        for frame in pipeline.run(str(input_path)):
            writer.write(draw_poses(frame))
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
    return frames_written


def main() -> int:
    parser = argparse.ArgumentParser(prog="padel-cv", description="Padel Vision CV pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Run the pipeline on a video")
    process.add_argument("input", type=Path, help="Input video path")
    process.add_argument("-o", "--output", type=Path, required=True, help="Annotated output path")
    process.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose model")
    process.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    process.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    args = parser.parse_args()
    if args.command == "process":
        process_video(args.input, args.output, args.model, args.conf, args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())

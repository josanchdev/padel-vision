"""Command-line entry point: process a video and write the annotated result."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from padel_cv.bounces import BallSample, detect_bounces
from padel_cv.match_data import MatchAnalysis
from padel_cv.pipeline import Pipeline, PipelineStage
from padel_cv.stages import (
    CourtDetectionStage,
    DummyShotStage,
    GroundTruthCourtStage,
    PlayerIdentityStage,
    PlayerPoseStage,
    StaticCourtStage,
)
from padel_cv.stages.pose import DEFAULT_TRACKER
from padel_cv.video_io import H264VideoWriter
from padel_cv.visualize import draw_ball, draw_poses, draw_shot_labels, overlay_minimap


@dataclass
class ProcessResult:
    frames_written: int
    shots_detected: int
    bounces_detected: int = 0


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
    static_court: str | None = None,
    start_frame: int = 0,
    on_progress: Callable[[float], None] | None = None,
    shot_model: str | None = None,
    ball_model: str | None = None,
    data_out: Path | None = None,
    write_video: bool = True,
) -> ProcessResult:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    capture.release()

    # Data extraction (ADR-0010) is separate from video rendering: with
    # write_video=False we skip the writer and all drawing for a fast data-only run.
    writer = H264VideoWriter(output_path, fps) if write_video else None
    analysis = MatchAnalysis(source_video=str(input_path), fps=fps)

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
    elif static_court is not None:
        coco_path, camera = static_court.rsplit(":", 1)
        court_stage = StaticCourtStage(coco_path, camera)
    elif homography_json is not None:
        court_stage = GroundTruthCourtStage(homography_json)
    if court_stage is not None:
        stages.append(court_stage)
        stages.append(PlayerIdentityStage())
    # Ball detection runs before shot detection so shots can use the ball
    # (ADR-0012). packages/ml imported lazily to keep cv free of torch.
    if ball_model is not None:
        from padel_ml.ball_stage import BallDetectionStage

        stages.append(BallDetectionStage(Path(ball_model)))
    if court_stage is not None:
        if shot_model is not None and ball_model is not None:
            # Best: detect shots from the ball's direction change + wrist (ADR-0012).
            from padel_ml.ball_shot_stage import BallShotStage

            stages.append(BallShotStage(Path(shot_model)))
        elif shot_model is not None:
            # No ball model: fall back to the wrist-speed proposal + classifier.
            from padel_ml.shot_stage import ClassifiedShotStage

            stages.append(ClassifiedShotStage(Path(shot_model)))
        else:
            stages.append(DummyShotStage())
    pipeline = Pipeline(stages)
    track_ids_seen: set[int] = set()
    active_shots: dict[int, tuple[int, str]] = {}
    shots_detected = 0
    ball_track: list[BallSample] = []
    shot_frames: set[int] = set()
    start = time.perf_counter()
    frames_written = 0
    try:
        for frame in pipeline.run(str(input_path), start_frame=start_frame):
            track_ids_seen.update(p.track_id for p in frame.poses if p.track_id is not None)
            shots_detected += len(frame.shot_events)
            shot_frames.update(e.frame_index for e in frame.shot_events)
            if frame.ball is not None:
                ball_track.append(
                    BallSample(frame.index, frame.ball.image_xy[0], frame.ball.image_xy[1])
                )
            analysis.accumulate_frame(frame)
            if writer is not None:
                canvas = draw_ball(draw_poses(frame), frame)
                canvas = draw_shot_labels(canvas, frame, active_shots)
                writer.write(overlay_minimap(canvas, frame))
            frames_written += 1
            if max_frames is not None and frames_written >= max_frames:
                break
            if frames_written % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"  {frames_written} frames ({frames_written / elapsed:.1f} fps)")
                if on_progress is not None and total_frames > 0:
                    on_progress(min(frames_written / total_frames, 1.0))
    finally:
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - start
    bounces = detect_bounces(ball_track, shot_frames) if ball_track else []
    analysis.set_bounces(bounces)
    if data_out is not None:
        analysis.write_json(data_out.with_suffix(".json"))
        analysis.write_csv(data_out.with_suffix(".csv"))
        print(f"Data written to {data_out.with_suffix('.json')} and .csv")
    if write_video:
        print(f"Wrote {frames_written} frames to {output_path} in {elapsed:.1f}s")
    else:
        print(f"Processed {frames_written} frames (data-only) in {elapsed:.1f}s")
    if track_ids_seen:
        print(f"Track IDs seen: {sorted(track_ids_seen)}")
    if shots_detected:
        print(f"Shots detected: {shots_detected}")
    if bounces:
        print(f"Bounces detected: {len(bounces)}")
    if on_progress is not None:
        on_progress(1.0)
    return ProcessResult(
        frames_written=frames_written,
        shots_detected=shots_detected,
        bounces_detected=len(bounces),
    )


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
    process.add_argument(
        "-o", "--output", type=Path, default=None, help="Annotated video output path"
    )
    process.add_argument(
        "--data-out",
        type=Path,
        default=None,
        help="Write structured data here (.json and .csv are appended); ADR-0010",
    )
    process.add_argument(
        "--no-video", action="store_true", help="Skip video rendering (fast data-only run)"
    )
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
    process.add_argument("--start", type=int, default=0, help="Start at this frame index")
    process.add_argument(
        "--static-court",
        default=None,
        metavar="COCO_JSON:CAMERA",
        help="Fixed-camera mode: exact homography from a one-time CVAT annotation (ADR-0006)",
    )
    process.add_argument(
        "--shot-model",
        default=None,
        help="PoseConv3D checkpoint: classify real shot types (else wrist-speed dummy)",
    )
    process.add_argument(
        "--ball-model",
        default=None,
        help="TrackNet checkpoint: detect the ball and its court position each frame",
    )

    extract = subparsers.add_parser(
        "extract-poses",
        help="Cache YOLO26-pose+tracking for a video (resumable shards)",
    )
    extract.add_argument("video", type=Path, help="Input video path")
    extract.add_argument("-o", "--cache-dir", type=Path, required=True, help="Shard output dir")
    extract.add_argument("--conf", type=float, default=0.3, help="Detection confidence")
    extract.add_argument("--imgsz", type=int, default=1920, help="Inference resolution")
    extract.add_argument("--device", default=None, help="Inference device, e.g. cuda or cpu")
    extract.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    ball = subparsers.add_parser(
        "extract-ball-frames",
        help="Cache downscaled frames + ball centres for TrackNet (resumable shards)",
    )
    ball.add_argument("video", type=Path, help="Match video path")
    ball.add_argument("--ball", type=Path, required=True, help="PadelTracker100 *_ball.json")
    ball.add_argument("-o", "--cache-dir", type=Path, required=True, help="Shard output dir")
    ball.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    build = subparsers.add_parser(
        "build-court-dataset",
        help="Auto-label court keypoints from PadelTracker100 GT homographies",
    )
    build.add_argument("video", type=Path, help="Match video path")
    build.add_argument("--homography", type=Path, required=True, help="GT homography JSON")
    build.add_argument("-o", "--output", type=Path, required=True, help="Dataset root dir")
    build.add_argument("--split", default="train", help="Dataset split (train/val)")
    build.add_argument("--every", type=int, default=150, help="Sample every N frames")

    evaluate = subparsers.add_parser(
        "eval-court", help="Evaluate a court model against GT homographies (error in meters)"
    )
    evaluate.add_argument("model", help="Trained court model weights")
    evaluate.add_argument("--homography", type=Path, required=True, help="GT homography JSON")
    evaluate.add_argument("--images", type=Path, required=True, help="Val images directory")
    evaluate.add_argument("--imgsz", type=int, default=1920, help="Inference resolution")

    sample = subparsers.add_parser(
        "sample-frames", help="Extract random frames from videos for annotation"
    )
    sample.add_argument("input_dir", type=Path, help="Directory containing videos")
    sample.add_argument("-o", "--output", type=Path, required=True, help="Output frames dir")
    sample.add_argument("--per-video", type=int, default=5, help="Frames sampled per video")
    sample.add_argument("--seed", type=int, default=42, help="Random seed (reproducible batches)")
    process.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    annot = subparsers.add_parser(
        "annotate-shots", help="Quick-mark shot annotator (ADR-0014): tap a key per shot"
    )
    annot.add_argument("video", type=Path, help="Video to annotate")
    annot.add_argument("-o", "--out", type=Path, required=True, help="Output shots CSV")
    annot.add_argument("--ball", type=Path, default=None, help="Ball detections JSON to overlay")
    annot.add_argument("--start", type=int, default=0, help="Start at this frame")
    annot.add_argument(
        "--end", type=int, default=None, help="Only keep audio candidates before this frame"
    )
    annot.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Audio/video file: auto-jump to hit-candidates (c/v) — much faster",
    )
    annot.add_argument(
        "--audio-threshold",
        type=float,
        default=0.5,
        help="Audio candidate sensitivity (higher = fewer, only clear pops)",
    )

    args = parser.parse_args()
    if args.command == "process":
        tracker = None if args.tracker == "none" else args.tracker
        write_video = not args.no_video
        if write_video and args.output is None:
            parser.error("--output is required unless --no-video is given")
        if not write_video and args.data_out is None:
            parser.error("--no-video needs --data-out (there would be no output otherwise)")
        process_video(
            args.input,
            args.output or Path("/dev/null"),
            args.model,
            args.conf,
            args.imgsz,
            args.max_frames,
            tracker,
            args.homography,
            args.court_model,
            args.static_court,
            args.start,
            shot_model=args.shot_model,
            ball_model=args.ball_model,
            data_out=args.data_out,
            write_video=write_video,
        )
    elif args.command == "extract-poses":
        from padel_cv.pose_cache import extract_poses_to_cache

        extract_poses_to_cache(
            args.video,
            args.cache_dir,
            args.conf,
            args.imgsz,
            max_frames=args.max_frames,
            device=args.device,
        )
    elif args.command == "extract-ball-frames":
        from padel_cv.ball_cache import extract_ball_frames_to_cache

        extract_ball_frames_to_cache(
            args.video, args.ball, args.cache_dir, max_frames=args.max_frames
        )
    elif args.command == "build-court-dataset":
        from padel_cv.datasets import build_court_dataset

        build_court_dataset(args.video, args.homography, args.output, args.split, args.every)
    elif args.command == "eval-court":
        from padel_cv.evaluation import evaluate_court_model

        evaluate_court_model(args.model, args.homography, args.images, args.imgsz)
    elif args.command == "sample-frames":
        sample_frames(args.input_dir, args.output, args.per_video, args.seed)
    elif args.command == "annotate-shots":
        import cv2

        from padel_cv.shot_annotator import _marks_summary, annotate

        candidates: list[int] | None = None
        if args.audio is not None:
            from padel_ml.audio_hits import hit_candidates

            cap = cv2.VideoCapture(str(args.video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            hits = hit_candidates(args.audio, threshold=args.audio_threshold)
            frames = [round(h.time_s * fps) for h in hits]
            end = args.end if args.end is not None else 10**12
            candidates = [f for f in frames if args.start <= f < end]
            print(f"{len(candidates)} candidatos de golpe por audio en el tramo (v = siguiente)")
        marks = annotate(
            args.video,
            args.out,
            ball_json=args.ball,
            start_frame=args.start,
            audio_candidates=candidates,
        )
        print(f"\n{len(marks)} golpes anotados -> {args.out}")
        print("por tipo:", _marks_summary(marks))
    return 0


if __name__ == "__main__":
    sys.exit(main())

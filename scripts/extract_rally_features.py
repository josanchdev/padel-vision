"""Extract pose + ball for every CVSPORTS rally, cached (ADR-0016).

This is the slow, one-off step the shot-type classifier needs: running YOLO pose
and TrackNet over all 99 rallies. Results are cached per rally, so a crash (the
WSL box does that) only costs the rally in flight.

Players are filtered with the hand-marked court mask, which is why this waits on
the court annotation: without it the tracker follows spectators and the
identities come out wrong.

    uv run python scripts/extract_rally_features.py [--limit N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball

from padel_cv.court_registry import court_file_for, load_corners
from padel_cv.pipeline import Frame, ImageArray
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage

REPO = Path(__file__).resolve().parents[1]
RALLIES = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel" / "rallies"
CACHE = REPO / "data" / "datasets" / "rally_features"
BALL_CKPT = REPO / "runs" / "ball_full" / "tracknetv3.pt"


def extract(video: Path, pose_stage: PlayerPoseStage, ball: BallDetector) -> dict[str, object]:
    """Pose (with stable player ids) + cleaned ball track for one rally."""
    court_json = court_file_for(video.stem)
    polygon = court_mask_polygon(load_corners(court_json)) if court_json else None

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    identity = PlayerIdentityTracker(fps=fps)
    # Keep the pose OBJECTS while scanning, not copies of their fields: the
    # identity tracker needs its first three seconds to work out who is who and
    # only then back-fills the ids onto those same objects. Reading player_id
    # inside the loop would freeze the rally's first 75 frames as unidentified —
    # and that is exactly where every serve lives.
    poses_by_frame: dict[int, list] = {}
    raw_ball: list[BallHit] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = pose_stage.process(
            Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
        )
        players = filter_players(frame.poses, polygon)
        identity.update(index, players)
        poses_by_frame[index] = players
        hit = ball.detect(image, index)
        if hit is not None:
            raw_ball.append(hit)
        index += 1
    capture.release()

    keypoints = {
        i: [p.keypoints.astype(np.float32) for p in players]
        for i, players in poses_by_frame.items()
    }
    player_ids = {
        i: [p.player_id if p.player_id is not None else -1 for p in players]
        for i, players in poses_by_frame.items()
    }
    boxes = {i: [p.bbox_xyxy for p in players] for i, players in poses_by_frame.items()}

    return {
        "keypoints": keypoints,
        "player_ids": player_ids,
        "boxes": boxes,
        "ball": {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(raw_ball)},
        "fps": fps,
        "n_frames": index,
        "court_json": str(court_json) if court_json else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rallies")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    videos = sorted(RALLIES.glob("*.mp4"))[: args.limit]
    pending = [v for v in videos if not (CACHE / f"{v.stem}.npz").exists()]
    cached = len(videos) - len(pending)
    print(f"{len(videos)} rallies · {cached} ya en cache · {len(pending)} por hacer")
    if not pending:
        return

    missing = [v.stem for v in pending if court_file_for(v.stem) is None]
    if missing:
        print(f"AVISO: {len(missing)} rallies sin pista marcada: {missing[:3]}")

    # conf 0.25, not the 0.4 default: measured over a full rally it finds all
    # four players in 85% of frames instead of 80%, and the far-side pair is
    # exactly who a higher threshold drops. yolo26n beats the larger 26m here
    # (85% vs 55%) and is twice as fast — the bottleneck is the tiny far-side
    # players, not model capacity.
    pose_stage = PlayerPoseStage(device=args.device, confidence=0.25)
    started = time.perf_counter()
    total_frames = 0
    for i, video in enumerate(pending, 1):
        rally_start = time.perf_counter()
        # A fresh detector per rally: its frame buffer must not span two videos.
        data = extract(video, pose_stage, BallDetector(BALL_CKPT, device=args.device))
        np.savez_compressed(
            CACHE / f"{video.stem}.npz",
            keypoints=np.array(data["keypoints"], dtype=object),
            player_ids=np.array(data["player_ids"], dtype=object),
            boxes=np.array(data["boxes"], dtype=object),
            ball=np.array(data["ball"], dtype=object),
            fps=data["fps"],
            n_frames=data["n_frames"],
            court_json=data["court_json"],
        )
        frames = int(cast(int, data["n_frames"]))
        total_frames += frames
        elapsed = time.perf_counter() - rally_start
        done_ratio = i / len(pending)
        eta = (time.perf_counter() - started) / done_ratio * (1 - done_ratio)
        print(
            f"  [{i:3d}/{len(pending)}] {video.stem:26s} {frames:5d} frames "
            f"({frames / elapsed:5.1f} fps)  ETA {eta / 60:.1f} min",
            flush=True,
        )
    total = time.perf_counter() - started
    print(f"\n{total_frames} frames en {total / 60:.1f} min ({total_frames / total:.1f} fps)")


if __name__ == "__main__":
    main()

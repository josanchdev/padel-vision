"""Extract pose + ball for every CVSPORTS rally, cached (ADR-0016).

This is the slow, one-off step the shot-type classifier needs: running YOLO pose
and TrackNet over all 99 rallies, with the system's own per-frame pass
(`track_rally`). Results are cached per rally, so a crash (the WSL box does that)
only costs the rally in flight.

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

import numpy as np
from padel_ml.ball_infer import BallDetector
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.rally_analysis import POSE_CONFIDENCE, track_rally

from padel_cv.court_registry import court_file_for
from padel_cv.paths import CVSPORTS_RALLIES, RALLY_FEATURES, RUNS
from padel_cv.pose import PoseDetector

BALL_CKPT = RUNS / "ball_full" / "tracknetv3.pt"


def extract(video: Path, pose_stage: PoseDetector, ball: BallDetector) -> dict[str, object]:
    """Pose (with stable player ids) + cleaned ball track for one rally."""
    court_json = court_file_for(video.stem)
    tracks = track_rally(video, pose_stage, ball, court_json)
    return {
        "keypoints": {
            i: [p.keypoints.astype(np.float32) for p in poses]
            for i, poses in tracks.players.items()
        },
        "player_ids": {i: [p.player_id for p in poses] for i, poses in tracks.players.items()},
        "boxes": {i: [p.bbox_xyxy for p in poses] for i, poses in tracks.players.items()},
        "ball": {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(tracks.raw_ball)},
        "fps": tracks.fps,
        "n_frames": tracks.n_frames,
        "court_json": str(court_json) if court_json else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rallies")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    args = parser.parse_args()

    RALLY_FEATURES.mkdir(parents=True, exist_ok=True)
    videos = sorted(CVSPORTS_RALLIES.glob("*.mp4"))[: args.limit]
    pending = [v for v in videos if not (RALLY_FEATURES / f"{v.stem}.npz").exists()]
    cached = len(videos) - len(pending)
    print(f"{len(videos)} rallies · {cached} ya en cache · {len(pending)} por hacer")
    if not pending:
        return

    missing = [v.stem for v in pending if court_file_for(v.stem) is None]
    if missing:
        print(f"AVISO: {len(missing)} rallies sin pista marcada: {missing[:3]}")

    # POSE_CONFIDENCE (0.25), not the 0.4 default: measured over a full rally it
    # finds all four players in 85% of frames instead of 80%, and the far-side
    # pair is exactly who a higher threshold drops. yolo26n beats the larger 26m
    # here (85% vs 55%) and is twice as fast — the bottleneck is the tiny
    # far-side players, not model capacity.
    pose_stage = PoseDetector(device=args.device, confidence=POSE_CONFIDENCE)
    started = time.perf_counter()
    total_frames = 0
    for i, video in enumerate(pending, 1):
        rally_start = time.perf_counter()
        # A fresh detector per rally: its frame buffer must not span two videos.
        data = extract(video, pose_stage, BallDetector(BALL_CKPT, device=args.device))
        np.savez_compressed(
            RALLY_FEATURES / f"{video.stem}.npz",
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

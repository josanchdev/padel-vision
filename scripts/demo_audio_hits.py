"""Demo: audio detects WHEN a hit happens, pose+ball decide WHO hit it.

Runs the whole backbone (ADR-0015) over one CVSPORTS rally and renders an
annotated video:
  - audio CRNN  -> hit windows (WHEN)
  - YOLO pose + court mask + stable 1-4 identity, TrackNet ball (cleaned)
  - weighted multi-frame voting (hit_assignment) -> the hitting player (WHO)
Each detected hit flashes "GOLPE" and highlights the hitting player's skeleton.

Usage:
    uv run python scripts/demo_audio_hits.py RALLY.mp4 -o demo.mp4 \
        --court data/datasets/vigo_court.json
If the audio checkpoint is missing, it is trained once (excluding this rally)
and saved.

This is a motivation/inspection tool, not part of the production pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.hit_assignment import FrameState, assign_hit_windows
from padel_ml.hit_assignment_eval import CORNER_INDICES

from padel_cv.pipeline import BallDetection, Frame, ImageArray, PoseDetection
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage
from padel_cv.visualize import COCO_SKELETON

_MIN_KP_CONF = 0.3
_FLASH_FRAMES = 12  # how long the "GOLPE" banner stays on after a hit
_REPO = Path(__file__).resolve().parents[1]

#: One stable colour per player slot, so a player keeps their colour all rally.
PLAYER_COLORS = {1: (80, 220, 80), 2: (255, 160, 0), 3: (60, 80, 255), 4: (0, 220, 255)}
HITTER_COLOR = (0, 0, 255)


def _draw_skeleton(
    img: np.ndarray, pose: PoseDetection, color: tuple[int, int, int], thick: int
) -> None:
    kp = pose.keypoints
    for a, b in COCO_SKELETON:
        if kp[a, 2] >= _MIN_KP_CONF and kp[b, 2] >= _MIN_KP_CONF:
            cv2.line(
                img,
                (int(kp[a, 0]), int(kp[a, 1])),
                (int(kp[b, 0]), int(kp[b, 1])),
                color,
                thick,
                cv2.LINE_AA,
            )
    for j in range(len(kp)):
        if kp[j, 2] >= _MIN_KP_CONF:
            cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), thick + 1, color, -1, cv2.LINE_AA)


def _label(img: np.ndarray, pose: PoseDetection, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, _, _ = pose.bbox_xyxy
    cv2.putText(
        img, text, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )


def _ensure_audio_ckpt(ckpt: Path, dataset_dir: Path, rally_name: str) -> Path:
    if ckpt.exists():
        return ckpt
    print(f"[audio] no checkpoint at {ckpt}; training (excluding {rally_name})...")
    from padel_ml.audio_train import fit_and_save

    cache = _REPO / "data" / "datasets" / "audio_cache.npz"
    return fit_and_save(dataset_dir, ckpt, cache=cache, exclude={rally_name})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rally", type=Path, help="CVSPORTS rally mp4")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--audio-ckpt", type=Path, default=_REPO / "runs" / "audio" / "audio_crnn.pt")
    ap.add_argument(
        "--ball-ckpt", type=Path, default=_REPO / "runs" / "ball_full" / "tracknetv3.pt"
    )
    ap.add_argument("--court", type=Path, default=_REPO / "data" / "datasets" / "vigo_court.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dataset_dir = _REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"

    # 1) WHEN: audio hit windows (seconds)
    ckpt = _ensure_audio_ckpt(args.audio_ckpt, dataset_dir, args.rally.name)
    from padel_ml.audio_train import detect_hit_windows_in_audio

    windows = detect_hit_windows_in_audio(args.rally, ckpt, device=args.device)
    print(f"[audio] {len(windows)} hits detected")

    cap = cv2.VideoCapture(str(args.rally))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    polygon = None
    if args.court.exists():
        corners = np.array(json.loads(args.court.read_text())["keypoints_px"])
        polygon = court_mask_polygon(corners[list(CORNER_INDICES), :2])

    # 2) per frame: pose + court mask + stable identity, and the ball
    pose_stage = PlayerPoseStage(device=args.device)
    ball = BallDetector(args.ball_ckpt, device=args.device)
    identity = PlayerIdentityTracker(fps=fps)
    frames: list[np.ndarray] = []
    poses_by_frame: dict[int, list[PoseDetection]] = {}
    raw_ball: list[BallHit] = []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        pf = pose_stage.process(
            Frame(index=idx, timestamp_s=idx / fps, image=cast(ImageArray, img))
        )
        players = filter_players(pf.poses, polygon)
        identity.update(idx, players)
        poses_by_frame[idx] = players
        hit = ball.detect(img, idx)
        if hit is not None:
            raw_ball.append(hit)
        frames.append(img)
        idx += 1
    cap.release()
    ball_by_frame = {
        b.frame_index: BallDetection((b.x_px, b.y_px), b.confidence)
        for b in postprocess_ball(raw_ball)
    }
    states = {
        i: FrameState(
            poses=[p for p in poses_by_frame.get(i, []) if p.player_id is not None],
            ball=ball_by_frame.get(i),
        )
        for i in range(idx)
    }
    print(f"[cv] {idx} frames, pose+ball+identity done")

    # 3) WHO: assign each audio hit window to a player
    assignment = assign_hit_windows(windows, states, fps)
    print(f"[assign] {assignment}")

    # 4) render
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
    flash_until = -1
    flash_player: int | None = None
    for i, img in enumerate(frames):
        if i in assignment:
            flash_until = i + _FLASH_FRAMES
            flash_player = assignment[i]
        hitter = flash_player if i <= flash_until else None
        for pose in poses_by_frame.get(i, []):
            if pose.player_id is None:
                continue
            is_hitter = hitter is not None and hitter > 0 and pose.player_id == hitter
            color = HITTER_COLOR if is_hitter else PLAYER_COLORS[pose.player_id]
            _draw_skeleton(img, pose, color, thick=4 if is_hitter else 2)
            _label(img, pose, f"J{pose.player_id}", color)
        ball_now = ball_by_frame.get(i)
        if ball_now is not None:
            cv2.circle(
                img,
                (int(ball_now.image_xy[0]), int(ball_now.image_xy[1])),
                9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if i <= flash_until:
            if flash_player is None:
                who = " (sin asignar)"
            elif flash_player < 0:
                who = f" (equipo {-flash_player})"
            else:
                who = f" - Jugador {flash_player}"
            cv2.putText(
                img,
                "GOLPE" + who,
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                HITTER_COLOR,
                4,
                cv2.LINE_AA,
            )
        writer.write(img)
    writer.release()
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

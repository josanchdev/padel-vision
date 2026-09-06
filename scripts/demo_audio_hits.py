"""Demo: audio detects WHEN a hit happens, pose+ball decide WHO hit it.

Runs the whole backbone (ADR-0015) over one CVSPORTS rally and renders an
annotated video:
  - audio CRNN  -> hit times (WHEN)
  - YOLO pose (tracked) + TrackNet ball, per video frame
  - weighted multi-frame voting (hit_assignment) -> the hitting track (WHO)
Each detected hit flashes "GOLPE" and highlights the hitting player's skeleton.

Usage:
    uv run python scripts/demo_audio_hits.py RALLY.mp4 -o demo.mp4 \
        --audio-ckpt runs/audio/audio_crnn.pt
If the audio checkpoint is missing, it is trained once (excluding this rally)
and saved.

This is a motivation/inspection tool, not part of the production pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from padel_ml.ball_infer import BallDetector
from padel_ml.hit_assignment import FrameState, assign_hits

from padel_cv.pipeline import BallDetection, Frame, PoseDetection
from padel_cv.stages.pose import PlayerPoseStage
from padel_cv.visualize import COCO_SKELETON, track_color

_MIN_KP_CONF = 0.3
_FLASH_FRAMES = 12  # how long the "GOLPE" banner stays on after a hit
_REPO = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dataset_dir = _REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"

    # 1) WHEN: audio hit times (seconds)
    ckpt = _ensure_audio_ckpt(args.audio_ckpt, dataset_dir, args.rally.name)
    from padel_ml.audio_train import detect_hits_in_audio

    hit_times = detect_hits_in_audio(args.rally, ckpt, device=args.device)
    print(f"[audio] {len(hit_times)} hits detected")

    cap = cv2.VideoCapture(str(args.rally))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hit_frames = sorted({round(t * fps) for t in hit_times})

    # 2) per frame: pose (tracked) + ball -> FrameState for assignment
    pose_stage = PlayerPoseStage(device=args.device)
    ball = BallDetector(args.ball_ckpt, device=args.device)
    frames: list[np.ndarray] = []
    states: dict[int, FrameState] = {}
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        pf = pose_stage.process(Frame(index=idx, timestamp_s=idx / fps, image=img))
        for p in pf.poses:  # use the tracker id as the player identity for the demo
            p.player_id = p.track_id
        bh = ball.detect(img, idx)
        bd = BallDetection((bh.x_px, bh.y_px), bh.confidence) if bh else None
        states[idx] = FrameState(poses=pf.poses, ball=bd)
        frames.append(img)
        idx += 1
    cap.release()
    print(f"[cv] {idx} frames, pose+ball done")

    # 3) WHO: assign each audio hit to a player track
    assignment = assign_hits(hit_frames, states)
    print(f"[assign] {assignment}")

    # 4) render
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
    flash_until = -1
    flash_player: int | None = None
    hit_set = set(hit_frames)
    for i, img in enumerate(frames):
        st = states[i]
        if i in hit_set:
            flash_until = i + _FLASH_FRAMES
            flash_player = assignment.get(i)
        hitter = flash_player if i <= flash_until else None
        for p in st.poses:
            is_hitter = hitter is not None and p.player_id == hitter
            color = (0, 0, 255) if is_hitter else track_color(p.track_id)
            _draw_skeleton(img, p, color, thick=4 if is_hitter else 2)
        if st.ball is not None:
            cv2.circle(
                img,
                (int(st.ball.image_xy[0]), int(st.ball.image_xy[1])),
                8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if i <= flash_until:
            who = f" (track {flash_player})" if flash_player is not None else " (sin asignar)"
            cv2.putText(
                img,
                "GOLPE" + who,
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
        writer.write(img)
    writer.release()
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

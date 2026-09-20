"""Render the whole pipeline over one rally: when, who, and what kind of shot.

Chains every trained piece so the result can be watched rather than read off a
table: the audio CRNN says WHEN a hit happens, pose + ball + weighted voting say
WHO hit it, and the BST-0 classifier says WHAT KIND of stroke it was.

Note on honesty: with a CVSPORTS rally the classifier has seen these hits during
training, so this is a "watch it work" demo, not a measurement. The measurement
is the cross-tournament 77.5% in docs/metrics.

    uv run python scripts/demo_full_pipeline.py RALLY.mp4 -o out.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.hit_assignment import FrameState, assign_hit
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, build_window
from padel_ml.shot_type_model import ShotTypeBST

from padel_cv.court_registry import court_file_for, load_corners
from padel_cv.pipeline import BallDetection, Frame, ImageArray, PoseDetection
from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage
from padel_cv.visualize import COCO_SKELETON

REPO = Path(__file__).resolve().parents[1]
BALL_CKPT = REPO / "runs" / "ball_full" / "tracknetv3.pt"
AUDIO_CKPT = REPO / "runs" / "audio" / "audio_crnn.pt"
TYPE_CKPT = REPO / "runs" / "shot_type" / "bst0.pt"

SPANISH = {"Forehand": "DERECHA", "Backhand": "REVES", "Smash": "REMATE", "Serve": "SAQUE"}
TYPE_COLOURS = {
    "Forehand": (235, 170, 60),
    "Backhand": (90, 160, 240),
    "Smash": (90, 90, 235),
    "Serve": (120, 200, 90),
}
PLAYER_COLOURS = {1: (80, 220, 80), 2: (255, 160, 0), 3: (60, 80, 255), 4: (0, 220, 255)}
REJECT = (150, 150, 150)
FLASH_FRAMES = 18
MIN_KP_CONF = 0.3


def draw_skeleton(
    image: np.ndarray, pose: PoseDetection, colour: tuple[int, int, int], thickness: int
) -> None:
    kp = pose.keypoints
    for a, b in COCO_SKELETON:
        if kp[a, 2] >= MIN_KP_CONF and kp[b, 2] >= MIN_KP_CONF:
            cv2.line(
                image,
                (int(kp[a, 0]), int(kp[a, 1])),
                (int(kp[b, 0]), int(kp[b, 1])),
                colour,
                thickness,
                cv2.LINE_AA,
            )
    for j in range(len(kp)):
        if kp[j, 2] >= MIN_KP_CONF:
            cv2.circle(
                image, (int(kp[j, 0]), int(kp[j, 1])), thickness + 1, colour, -1, cv2.LINE_AA
            )


def text(
    image: np.ndarray,
    string: str,
    org: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    weight: int = 2,
) -> None:
    cv2.putText(
        image, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 3, cv2.LINE_AA
    )
    cv2.putText(image, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rally", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Reject below this confidence (0 = never reject)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # 1) WHEN — the audio detector
    from padel_ml.audio_train import detect_hits_in_audio

    hit_times = detect_hits_in_audio(args.rally, AUDIO_CKPT, device=device)
    capture = cv2.VideoCapture(str(args.rally))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hit_frames = sorted({round(t * fps) for t in hit_times})
    print(f"[audio] {len(hit_frames)} golpes detectados")

    # 2) pose + ball + stable identities
    court = court_file_for(args.rally.stem)
    polygon = court_mask_polygon(load_corners(court)) if court else None
    pose_stage = PlayerPoseStage(device=device, confidence=0.25)
    ball_detector = BallDetector(BALL_CKPT, device=device)
    identity = PlayerIdentityTracker(fps=fps)
    frames: list[np.ndarray] = []
    poses_by_frame: dict[int, list[PoseDetection]] = {}
    raw_ball: list[BallHit] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        processed = pose_stage.process(
            Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
        )
        players = filter_players(processed.poses, polygon)
        identity.update(index, players)
        poses_by_frame[index] = players
        hit = ball_detector.detect(image, index)
        if hit is not None:
            raw_ball.append(hit)
        frames.append(image)
        index += 1
    capture.release()
    ball_by_frame = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(raw_ball)}
    coverage = 100 * len(ball_by_frame) / max(index, 1)
    print(f"[cv] {index} frames · pelota en {len(ball_by_frame)} ({coverage:.0f}%)")

    keypoints_by_frame = {
        i: [p.keypoints.astype(np.float32) for p in players]
        for i, players in poses_by_frame.items()
    }
    player_ids_by_frame = {
        i: [p.player_id if p.player_id is not None else -1 for p in players]
        for i, players in poses_by_frame.items()
    }

    # 3) WHO — weighted multi-frame vote
    states = {
        i: FrameState(
            poses=[p for p in players if p.player_id is not None],
            ball=BallDetection(ball_by_frame[i], 1.0) if i in ball_by_frame else None,
        )
        for i, players in poses_by_frame.items()
    }
    hitters = {frame: assign_hit(frame, states) for frame in hit_frames}

    # 4) WHAT — the shot-type classifier
    checkpoint = torch.load(TYPE_CKPT, map_location=device, weights_only=False)
    model = ShotTypeBST(seq_len=checkpoint["seq_len"], n_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    shots: dict[int, tuple[int | None, str | None, float]] = {}
    for position, frame in enumerate(hit_frames):
        hitter = hitters.get(frame)
        label, confidence = None, 0.0
        if hitter is not None and hitter > 0:
            window = build_window(
                keypoints_by_frame,
                player_ids_by_frame,
                ball_by_frame,
                hit_frames,
                position,
                hitter,
                fps,
                0,
                args.rally.stem,
                "",
                SEQ_LEN,
            )
            if window is not None:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(window.pose)[None].to(device),
                        torch.from_numpy(window.ball)[None].to(device),
                    )
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                best = int(np.argmax(probabilities))
                confidence = float(probabilities[best])
                label = CLASSES[best] if confidence >= args.threshold else None
        shots[frame] = (hitter, label, confidence)
        name = SPANISH.get(label or "", "SIN CLASIFICAR")
        print(f"  golpe f={frame:5d}  J{hitter}  {name:14s} ({confidence:.2f})")

    # 5) render
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    flash_until, current = -1, None
    counts: dict[str, int] = {}
    for i, image in enumerate(frames):
        if i in shots:
            flash_until, current = i + FLASH_FRAMES, shots[i]
            if current[1]:
                counts[current[1]] = counts.get(current[1], 0) + 1
        active = current if i <= flash_until else None
        hitter = active[0] if active else None

        for pose in poses_by_frame.get(i, []):
            if pose.player_id is None:
                continue
            is_hitter = hitter is not None and pose.player_id == hitter
            colour = (
                TYPE_COLOURS.get(active[1] or "", REJECT)
                if is_hitter and active
                else PLAYER_COLOURS[pose.player_id]
            )
            draw_skeleton(image, pose, colour, 4 if is_hitter else 2)
            x1, y1 = pose.bbox_xyxy[0], pose.bbox_xyxy[1]
            text(image, f"J{pose.player_id}", (int(x1), int(y1) - 8), 0.7, colour)

        if i in ball_by_frame:
            x, y = ball_by_frame[i]
            cv2.circle(image, (int(x), int(y)), 9, (0, 255, 255), 2, cv2.LINE_AA)

        if active is not None:
            label = SPANISH.get(active[1] or "", "SIN CLASIFICAR")
            colour = TYPE_COLOURS.get(active[1] or "", REJECT)
            who = f"J{active[0]}" if active[0] else "?"
            text(image, f"{label}  ·  {who}  ·  {active[2]:.0%}", (40, 90), 1.5, colour, 3)

        # Running tally, laid out upwards from a fixed baseline so it never
        # runs off the bottom of the frame as classes appear.
        rows = sorted(counts.items(), key=lambda kv: -kv[1])
        baseline = height - 40
        for position, (name, count) in enumerate(reversed(rows)):
            text(
                image,
                f"{SPANISH[name]}: {count}",
                (40, baseline - position * 28),
                0.6,
                TYPE_COLOURS[name],
            )
        text(
            image,
            f"{sum(counts.values())} golpes",
            (40, baseline - len(rows) * 28 - 6),
            0.7,
            (235, 235, 235),
        )
        writer.write(image)
    writer.release()
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()

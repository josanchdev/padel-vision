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
from padel_ml.ball_trajectory import clean_track
from padel_ml.hit_assignment import FrameState, assign_hit
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, build_window
from padel_ml.shot_type_model import ShotTypeBST
from padel_ml.shot_type_train import SERVE_INDEX

from padel_cv import overlay
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
#: BGR. Chosen to stay distinct against a blue court and from each other.
TYPE_COLOURS = {
    "Forehand": (60, 200, 255),  # amber
    "Backhand": (255, 190, 90),  # cyan-blue
    "Smash": (90, 90, 255),  # red
    "Serve": (120, 230, 120),  # green
}
PLAYER_COLOURS = {1: (80, 220, 80), 2: (255, 160, 0), 3: (60, 80, 255), 4: (0, 220, 255)}
REJECT = (150, 150, 150)
FLASH_FRAMES = 18
MIN_KP_CONF = 0.3


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
    # Two tracks on purpose. Assignment votes on the plain post-processed one:
    # measured against the paper's ground truth the parabolic smoothing costs it
    # about a point, because a fit that spans a hit rounds off the very moment
    # the vote depends on. Drawing uses the smoothed one, whose jerk is 23 px ->
    # under 7 and reads as a trajectory rather than a scatter of dots.
    ball_by_frame = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(raw_ball)}
    ball_for_drawing = {p.frame_index: (p.x_px, p.y_px) for p in clean_track(raw_ball)}
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
                # Only the rally's opening hit may be a serve (ADR-0016): every
                # serve in the dataset is one, and letting the model call a
                # mid-rally hit a serve was its single biggest source of false
                # positives (precision 0.722 -> 1.000 once forbidden).
                if best == SERVE_INDEX and position > 0:
                    without_serve = probabilities.copy()
                    without_serve[SERVE_INDEX] = -1.0
                    best = int(np.argmax(without_serve))
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

        hitter_pose = None
        for pose in poses_by_frame.get(i, []):
            if pose.player_id is None:
                continue
            is_hitter = hitter is not None and pose.player_id == hitter
            if is_hitter:
                hitter_pose = pose
                continue  # drawn last, so it sits on top of the others
            overlay.skeleton(image, pose.keypoints, COCO_SKELETON, PLAYER_COLOURS[pose.player_id])
            overlay.label(
                image,
                f"J{pose.player_id}",
                (int(pose.bbox_xyxy[0]), int(pose.bbox_xyxy[1]) - 6),
                scale=0.5,
                accent=PLAYER_COLOURS[pose.player_id],
            )

        overlay.ball_trail(image, [ball_for_drawing.get(f) for f in range(max(i - 11, 0), i + 1)])

        if hitter_pose is not None and active is not None:
            colour = TYPE_COLOURS.get(active[1] or "", REJECT)
            # The hitter keeps HIS OWN colour, only drawn heavier: recolouring him
            # by stroke type would break the one thing the colour is for — telling
            # the four players apart — exactly when the viewer is looking hardest.
            # The stroke type is carried by the label instead.
            player_colour = PLAYER_COLOURS[hitter_pose.player_id]
            overlay.skeleton(
                image,
                hitter_pose.keypoints,
                COCO_SKELETON,
                player_colour,
                thickness=3,
                joint_radius=4,
            )
            # The verdict goes right above the player who produced it: a tag in
            # the corner makes the viewer hunt for who it refers to.
            x1, y1, x2, _ = hitter_pose.bbox_xyxy
            name = SPANISH.get(active[1] or "", "SIN CLASIFICAR")
            # Clear of the head: the box top already sits at the crown, so the
            # plate would otherwise cover the face of the player it describes.
            top = overlay.label(
                image,
                f"{name}  {active[2]:.0%}",
                (int((x1 + x2) / 2), int(y1) - 34),
                scale=0.72,
                accent=colour,
                centred=True,
            )
            overlay.label(
                image,
                f"JUGADOR {active[0]}",
                (int((x1 + x2) / 2), top[1] - 4),
                scale=0.46,
                accent=player_colour,  # same colour as his skeleton
                centred=True,
            )

        rows = [
            (SPANISH[name], str(counts.get(name, 0)), TYPE_COLOURS[name])
            for name in CLASSES
            if counts.get(name)
        ]
        if rows:
            overlay.panel(image, f"GOLPES DETECTADOS   {sum(counts.values())}", rows)
        writer.write(image)
    writer.release()
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()

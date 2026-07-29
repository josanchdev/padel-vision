"""Drawing helpers: skeleton overlay and top-down court minimap."""

from __future__ import annotations

import cv2
import numpy as np

from padel_cv.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SERVICE_LINE_FROM_NET_M
from padel_cv.pipeline import Frame, ImageArray, PoseDetection

# COCO-17 skeleton: pairs of keypoint indices to connect with a line.
# 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
# 11-12 hips, 13-14 knees, 15-16 ankles.
COCO_SKELETON: list[tuple[int, int]] = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11),
    (6, 12),  # torso
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

BONE_COLOR = (0, 200, 255)
JOINT_COLOR = (255, 120, 0)
BOX_COLOR = (80, 220, 80)
MIN_KEYPOINT_CONFIDENCE = 0.3

# Distinct BGR colors assigned to track IDs so each tracked player keeps a
# stable color for as long as their ID survives.
TRACK_COLORS: list[tuple[int, int, int]] = [
    (80, 220, 80),  # green
    (60, 80, 255),  # red
    (255, 160, 0),  # blue
    (0, 220, 255),  # yellow
    (255, 0, 200),  # magenta
    (200, 255, 0),  # cyan
]


def track_color(track_id: int | None) -> tuple[int, int, int]:
    """Deterministic color for a track ID; default box color for untracked."""
    if track_id is None:
        return BOX_COLOR
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def pose_color_and_label(pose: PoseDetection) -> tuple[tuple[int, int, int], str]:
    """Stable player slots win over volatile track IDs for color and label."""
    if pose.player_id is not None:
        return TRACK_COLORS[(pose.player_id - 1) % len(TRACK_COLORS)], f"J{pose.player_id}"
    if pose.track_id is not None:
        return track_color(pose.track_id), f"#{pose.track_id}"
    return BOX_COLOR, f"{pose.confidence:.2f}"


BALL_COLOR = (0, 255, 255)  # yellow, like a padel ball


def draw_ball(canvas: ImageArray, frame: Frame) -> ImageArray:
    """Mark the detected ball with a small circle on the annotated frame."""
    if frame.ball is None:
        return canvas
    x, y = (int(v) for v in frame.ball.image_xy)
    cv2.circle(canvas, (x, y), 6, BALL_COLOR, 2)
    cv2.circle(canvas, (x, y), 1, BALL_COLOR, -1)
    return canvas


def draw_poses(frame: Frame) -> ImageArray:
    """Return a copy of the frame image with boxes and skeletons drawn."""
    canvas = frame.image.copy()
    for pose in frame.poses:
        x1, y1, x2, y2 = (int(v) for v in pose.bbox_xyxy)
        if pose.on_court is False:
            # Off-court people (spectators, staff): thin grey box, no skeleton.
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (128, 128, 128), 1)
            continue
        color, label = pose_color_and_label(pose)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for a, b in COCO_SKELETON:
            if (
                pose.keypoints[a, 2] >= MIN_KEYPOINT_CONFIDENCE
                and pose.keypoints[b, 2] >= MIN_KEYPOINT_CONFIDENCE
            ):
                pt_a = (int(pose.keypoints[a, 0]), int(pose.keypoints[a, 1]))
                pt_b = (int(pose.keypoints[b, 0]), int(pose.keypoints[b, 1]))
                cv2.line(canvas, pt_a, pt_b, BONE_COLOR, 2)
        for x, y, kp_conf in pose.keypoints:
            if kp_conf >= MIN_KEYPOINT_CONFIDENCE:
                cv2.circle(canvas, (int(x), int(y)), 3, JOINT_COLOR, -1)
    return canvas


MINIMAP_HEIGHT_PX = 400
MINIMAP_MARGIN_PX = 20
COURT_FLOOR_COLOR = (140, 90, 30)
COURT_LINE_COLOR = (255, 255, 255)


def draw_minimap(frame: Frame, height_px: int = MINIMAP_HEIGHT_PX) -> ImageArray:
    """Top-down court view with on-court players as colored dots."""
    scale = height_px / COURT_LENGTH_M
    width_px = int(COURT_WIDTH_M * scale)
    canvas = np.full((height_px, width_px, 3), COURT_FLOOR_COLOR, dtype=np.uint8)

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        # Flip both axes so the minimap matches the viewer's frame: the court's
        # near side (y=0) at the BOTTOM, and its left side (x=0 = corner_near_left)
        # on the RIGHT — the broadcast camera mirrors the court left/right.
        return int((COURT_WIDTH_M - x_m) * scale), int((COURT_LENGTH_M - y_m) * scale)

    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), COURT_LINE_COLOR, 2)
    net_y = to_px(0, NET_Y_M)[1]
    cv2.line(canvas, (0, net_y), (width_px, net_y), COURT_LINE_COLOR, 3)
    for service_y_m in (NET_Y_M - SERVICE_LINE_FROM_NET_M, NET_Y_M + SERVICE_LINE_FROM_NET_M):
        service_y = to_px(0, service_y_m)[1]
        cv2.line(canvas, (0, service_y), (width_px, service_y), COURT_LINE_COLOR, 1)
        center_x = to_px(COURT_WIDTH_M / 2, 0)[0]
        cv2.line(
            canvas,
            (center_x, min(net_y, service_y)),
            (center_x, max(net_y, service_y)),
            COURT_LINE_COLOR,
            1,
        )

    for pose in frame.poses:
        if pose.court_position_m is None or not pose.on_court:
            continue
        x, y = to_px(*pose.court_position_m)
        color, label = pose_color_and_label(pose)
        cv2.circle(canvas, (x, y), 8, color, -1)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), 1)
        cv2.putText(canvas, label, (x - 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    if frame.ball is not None and frame.ball.court_xy_m is not None:
        bx, by = to_px(*frame.ball.court_xy_m)
        cv2.circle(canvas, (bx, by), 4, BALL_COLOR, -1)
        cv2.circle(canvas, (bx, by), 4, (0, 0, 0), 1)
    return canvas


def overlay_minimap(canvas: ImageArray, frame: Frame) -> ImageArray:
    """Blend the minimap into the bottom-right corner of an annotated frame."""
    if frame.homography is None:
        return canvas
    minimap = draw_minimap(frame)
    map_h, map_w = minimap.shape[:2]
    img_h, img_w = canvas.shape[:2]
    y0 = img_h - map_h - MINIMAP_MARGIN_PX
    x0 = img_w - map_w - MINIMAP_MARGIN_PX
    region = canvas[y0 : y0 + map_h, x0 : x0 + map_w]
    canvas[y0 : y0 + map_h, x0 : x0 + map_w] = cv2.addWeighted(minimap, 0.85, region, 0.15, 0)
    return canvas


SHOT_FLASH_FRAMES = 15

# Spanish shot-type labels for the overlay; NoShot is filtered upstream.
_SHOT_LABEL_ES = {
    "Forehand": "Derecha",
    "Backhand": "Reves",
    "Smash": "Remate",
    "Serve": "Saque",
    "Other": "Otro",
}


def draw_shot_labels(
    canvas: ImageArray, frame: Frame, active_shots: dict[int, tuple[int, str]]
) -> ImageArray:
    """Flash the shot type over a player who just hit.

    active_shots maps player_id -> (remaining frames, label to show).
    """
    for event in frame.shot_events:
        label = _SHOT_LABEL_ES.get(event.label, event.label)
        active_shots[event.player_id] = (SHOT_FLASH_FRAMES, label)
    for pose in frame.poses:
        state = active_shots.get(pose.player_id) if pose.player_id is not None else None
        if state is None or state[0] <= 0:
            continue
        x1, y1, _, _ = (int(v) for v in pose.bbox_xyxy)
        cv2.putText(
            canvas,
            f"J{pose.player_id}: {state[1]}",
            (x1 - 10, max(y1 - 34, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
    for player_id in list(active_shots):
        ttl, label = active_shots[player_id]
        if ttl <= 1:
            del active_shots[player_id]
        else:
            active_shots[player_id] = (ttl - 1, label)
    return canvas

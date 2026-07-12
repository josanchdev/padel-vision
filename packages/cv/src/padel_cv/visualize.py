"""Drawing helpers: skeleton overlay on frames."""

from __future__ import annotations

import cv2

from padel_cv.pipeline import Frame, ImageArray

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


def draw_poses(frame: Frame) -> ImageArray:
    """Return a copy of the frame image with boxes and skeletons drawn."""
    canvas = frame.image.copy()
    for pose in frame.poses:
        x1, y1, x2, y2 = (int(v) for v in pose.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{pose.confidence:.2f}" if pose.track_id is None else f"#{pose.track_id}"
        cv2.putText(canvas, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
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

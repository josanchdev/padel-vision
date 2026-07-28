"""Turn ball detections into a verification video and CVAT pre-annotations.

The verification mp4 marks each detected ball so the model can be eyeballed on
new footage (Phase 1d). The CVAT export writes the detections as COCO boxes —
CVAT imports COCO natively, and PadelTracker100's ball labels are already COCO
boxes, so corrections stay in one consistent format (Phase 2 pre-labeling).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from padel_cv.video_io import H264VideoWriter
from padel_ml.ball_infer import BallHit, detect_ball_in_video

BALL_COLOR = (0, 255, 255)  # yellow
BOX_HALF = 6  # half-size of the exported ball box (px); the ball is ~8px


def write_verification_video(
    video_path: Path,
    checkpoint: Path,
    output_path: Path,
    min_confidence: float = 0.5,
    max_frames: int | None = None,
) -> list[BallHit]:
    """Write an mp4 with each detected ball circled; return the detections."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = H264VideoWriter(output_path, fps)

    def overlay(_index: int, image: np.ndarray, hit: BallHit | None) -> None:
        if hit is not None:
            cv2.circle(image, (int(hit.x_px), int(hit.y_px)), 7, BALL_COLOR, 2)
        writer.write(image)

    try:
        hits = detect_ball_in_video(
            video_path, checkpoint, min_confidence, max_frames, on_frame=overlay
        )
    finally:
        writer.release()
    return hits


def export_cvat_coco(
    hits: list[BallHit],
    video_path: Path,
    output_json: Path,
    n_frames: int,
    frame_size: tuple[int, int],
) -> None:
    """Write detections as a COCO json CVAT can import as ball-box proposals.

    One image entry per video frame (CVAT names them frame_000000.PNG...), one
    ball annotation per detection. frame_size is (width, height).
    """
    width, height = frame_size
    images = [
        {"id": i + 1, "file_name": f"frame_{i:06d}.PNG", "width": width, "height": height}
        for i in range(n_frames)
    ]
    annotations = [
        {
            "id": j + 1,
            "image_id": hit.frame_index + 1,
            "category_id": 1,
            "bbox": [hit.x_px - BOX_HALF, hit.y_px - BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF],
            "area": (2 * BOX_HALF) ** 2,
            "iscrowd": 0,
            "score": round(hit.confidence, 3),
        }
        for j, hit in enumerate(hits)
    ]
    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"description": f"Ball pre-annotations for {video_path.name}"},
        "categories": [{"id": 1, "name": "Ball", "supercategory": ""}],
        "images": images,
        "annotations": annotations,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco))

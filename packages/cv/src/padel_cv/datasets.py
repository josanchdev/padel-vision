"""Court-keypoint dataset generation.

Bootstraps training data for the court detector (ADR-0001) without manual
annotation: PadelTracker100 ships a ground-truth homography per frame, and the
13 schema points have known court coordinates, so projecting them through the
inverse homography yields their pixel locations in every frame for free.

Labels are written in Ultralytics YOLO-pose format: one "court" object per
image whose bounding box encloses the visible keypoints. Points that fall
outside the image get visibility 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from padel_cv.court import COURT_KEYPOINTS_M, HomographyArray


def project_court_points_to_pixels(
    homography_px_to_m: HomographyArray,
) -> np.ndarray:
    """Pixel positions of the 13 schema points given a px->m homography."""
    m_to_px = np.linalg.inv(homography_px_to_m)
    points = np.concatenate([COURT_KEYPOINTS_M, np.ones((len(COURT_KEYPOINTS_M), 1))], axis=1)
    projected = (m_to_px @ points.T).T
    return np.asarray(projected[:, :2] / projected[:, 2:3])


def yolo_pose_label(
    keypoints_px: np.ndarray, width: int, height: int, margin_px: float = 8.0
) -> str | None:
    """One-line YOLO-pose label for the court object, or None if too few points."""
    in_bounds = (
        (keypoints_px[:, 0] >= -margin_px)
        & (keypoints_px[:, 0] < width + margin_px)
        & (keypoints_px[:, 1] >= -margin_px)
        & (keypoints_px[:, 1] < height + margin_px)
    )
    if in_bounds.sum() < 4:
        return None
    visible = keypoints_px[in_bounds]
    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)
    x1, y1 = max(float(x1), 0.0), max(float(y1), 0.0)
    x2, y2 = min(float(x2), width - 1.0), min(float(y2), height - 1.0)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for (x, y), ok in zip(keypoints_px, in_bounds, strict=True):
        if ok:
            parts.append(f"{x / width:.6f} {y / height:.6f} 2")
        else:
            parts.append("0 0 0")
    return " ".join(parts)


def build_court_dataset(
    video_path: Path,
    homography_json: Path,
    output_dir: Path,
    split: str,
    every_n_frames: int = 150,
) -> int:
    """Sample frames from a PadelTracker100 video and write YOLO-pose labels."""
    with open(homography_json) as f:
        entries = json.load(f)["homography"]
    h_by_image_id: dict[int, HomographyArray] = {
        e["image_id"]: np.array(e["H"], dtype=np.float64) for e in entries
    }

    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    for frame_index in range(0, total, every_n_frames):
        homography = h_by_image_id.get(frame_index + 1)
        if homography is None:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok:
            continue
        height, width = image.shape[:2]
        label = yolo_pose_label(project_court_points_to_pixels(homography), width, height)
        if label is None:
            continue
        stem = f"{video_path.stem}_{frame_index:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (labels_dir / f"{stem}.txt").write_text(label + "\n")
        written += 1
    capture.release()
    print(f"{video_path.name} -> {written} labeled frames in {images_dir}")
    return written

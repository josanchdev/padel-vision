"""Formal evaluation of the court detector against GT homographies.

Metric: for each val frame, sample a grid of world points covering the court,
obtain their pixel locations through the GT homography inverse, project those
pixels back to meters with the homography estimated from detected keypoints,
and measure the distances. This captures the end-to-end effect the detector's
error has on projected player positions, in meters.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from padel_cv.court import homography_from_keypoints


def evaluate_court_model(
    model_path: str,
    homography_json: Path,
    val_images_dir: Path,
    image_size: int = 1920,
) -> dict[str, float]:
    """Returns error stats in meters plus homography coverage on val frames."""
    from ultralytics import YOLO

    with open(homography_json) as f:
        gt_by_id = {e["image_id"]: np.array(e["H"]) for e in json.load(f)["homography"]}

    model = YOLO(model_path)
    grid_m = np.array(
        [[x, y] for x in np.linspace(0.5, 9.5, 8) for y in np.linspace(0.5, 19.5, 12)]
    )
    ones = np.ones((len(grid_m), 1))

    errors: list[float] = []
    rejected = 0
    for img_path in sorted(val_images_dir.glob("*.jpg")):
        frame_index = int(img_path.stem.rsplit("_", 1)[1])
        gt = gt_by_id.get(frame_index + 1)
        if gt is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        result = model.predict(image, conf=0.3, imgsz=image_size, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            rejected += 1
            continue
        best = int(result.boxes.conf.argmax())
        keypoints = result.keypoints.data[best].cpu().numpy().astype(np.float32)
        predicted = homography_from_keypoints(keypoints)
        if predicted is None:
            rejected += 1
            continue
        pixels = (np.linalg.inv(gt) @ np.concatenate([grid_m, ones], 1).T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        back = (predicted @ np.concatenate([pixels, ones], 1).T).T
        back = back[:, :2] / back[:, 2:3]
        # Court orientation (which end is y=0) is arbitrary and both are valid
        # homographies: score against the GT convention and its 180° rotation.
        error_direct = np.linalg.norm(back - grid_m, axis=1).mean()
        rotated = np.stack([10.0 - back[:, 0], 20.0 - back[:, 1]], axis=1)
        error_rotated = np.linalg.norm(rotated - grid_m, axis=1).mean()
        errors.append(float(min(error_direct, error_rotated)))

    stats = {
        "frames": float(len(errors)),
        "rejected": float(rejected),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }
    print(
        f"frames: {stats['frames']:.0f} | sin homografía: {stats['rejected']:.0f} | "
        f"error medio: {stats['mean_m']:.3f} m | mediana: {stats['median_m']:.3f} m | "
        f"p95: {stats['p95_m']:.3f} m | max: {stats['max_m']:.3f} m"
    )
    return stats

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


def evaluate_shot_detection(
    pose_json: Path,
    shots_csv: Path,
    speed_threshold: float = 0.6,
    match_margin_frames: int = 5,
    max_frames: int | None = None,
) -> dict[str, float]:
    """Run the wrist-speed heuristic on GT skeletons vs GT shot labels.

    Uses PadelTracker100's refined pose annotations as input (isolating the
    heuristic from our pose detector) and its per-frame shot ranges as ground
    truth. GT events are contiguous has_shot runs; a prediction inside a run
    (with margin) is a hit. Reports event-level precision/recall/F1.
    """
    import csv

    from padel_cv.pipeline import Frame, PoseDetection
    from padel_cv.stages import DummyShotStage

    with open(pose_json) as f:
        coco = json.load(f)
    frame_of_image = {
        img["id"]: int(Path(img["file_name"]).stem.rsplit("_", 1)[1]) for img in coco["images"]
    }
    poses_by_frame: dict[int, list[np.ndarray]] = {}
    for ann in coco["annotations"]:
        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        kp[:, 2] = np.where(kp[:, 2] > 0, 0.9, 0.0)  # COCO visibility -> confidence
        poses_by_frame.setdefault(frame_of_image[ann["image_id"]], []).append(kp)

    # GT shot events: contiguous runs of has_shot=1
    gt_events: list[tuple[int, int]] = []
    with open(shots_csv) as f:
        run_start: int | None = None
        prev = -1
        for row in csv.DictReader(f, delimiter=";"):
            idx = int(Path(row["file_name"]).stem.rsplit("_", 1)[1])
            has_shot = row["has_shot"] == "1"
            if has_shot and run_start is None:
                run_start = idx
            elif not has_shot and run_start is not None:
                gt_events.append((run_start, prev))
                run_start = None
            prev = idx
            if max_frames is not None and idx >= max_frames:
                break
        if run_start is not None and prev >= 0:
            gt_events.append((run_start, prev))

    # Pseudo-tracking of GT skeletons by nearest hip center across frames.
    stage = DummyShotStage(speed_threshold=speed_threshold)
    predictions: list[int] = []
    previous: dict[int, np.ndarray] = {}
    last_index = max(poses_by_frame) if max_frames is None else min(max(poses_by_frame), max_frames)
    for frame_index in range(last_index + 1):
        skeletons = poses_by_frame.get(frame_index, [])
        centers = [kp[[11, 12], :2].mean(axis=0) for kp in skeletons]
        assigned: dict[int, np.ndarray] = {}
        used: set[int] = set()
        for pid, prev_center in previous.items():
            best: int | None = None
            best_d = 1e9
            for i, c in enumerate(centers):
                d = float(np.linalg.norm(c - prev_center))
                if i not in used and d < best_d:
                    best, best_d = i, d
            if best is not None and best_d < 150:
                assigned[pid] = centers[best]
                used.add(best)
        free_ids = [i for i in range(1, 5) if i not in assigned]
        for i, c in enumerate(centers):
            if i not in used and free_ids:
                assigned[free_ids.pop(0)] = c
                used.add(i)
        frame = Frame(index=frame_index, timestamp_s=0.0, image=np.zeros((1, 1, 3), np.uint8))
        center_to_id = {tuple(np.round(c, 1)): pid for pid, c in assigned.items()}
        for kp in skeletons:
            center = tuple(np.round(kp[[11, 12], :2].mean(axis=0), 1))
            skeleton_pid = center_to_id.get(center)
            if skeleton_pid is None:
                continue
            frame.poses.append(
                PoseDetection(
                    bbox_xyxy=(0, 0, 1, 1),
                    confidence=0.9,
                    keypoints=kp,
                    player_id=skeleton_pid,
                )
            )
        frame = stage.process(frame)
        predictions.extend(e.frame_index for e in frame.shot_events)
        previous = assigned

    hits = 0
    matched: set[int] = set()
    for p in predictions:
        for i, (start, end) in enumerate(gt_events):
            if start - match_margin_frames <= p <= end + match_margin_frames:
                hits += 1
                matched.add(i)
                break
    precision = hits / len(predictions) if predictions else 0.0
    recall = len(matched) / len(gt_events) if gt_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    stats = {
        "gt_events": float(len(gt_events)),
        "predictions": float(len(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(
        f"umbral={speed_threshold}: eventos GT={len(gt_events)}, predicciones={len(predictions)}, "
        f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}"
    )
    return stats

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
from typing import Any

import cv2
import numpy as np

from padel_cv.court import COURT_KEYPOINTS_M, HomographyArray

# The court is symmetric under a 180° rotation (x -> 10-x, y -> 20-y), which
# reverses the keypoint order. Annotation convention: "near" is the half
# closest to the camera (bottom of the image); GT homographies may use either
# end of the court as y=0, so auto-generated labels must be normalized.
ROT180_IDX: list[int] = list(range(12, -1, -1))


def project_court_points_to_pixels(
    homography_px_to_m: HomographyArray,
) -> np.ndarray:
    """Pixel positions of the 13 schema points given a px->m homography.

    Point order is normalized so that the "near" points are the ones at the
    bottom of the image (camera-relative convention used by annotators).
    """
    m_to_px = np.linalg.inv(homography_px_to_m)
    points = np.concatenate([COURT_KEYPOINTS_M, np.ones((len(COURT_KEYPOINTS_M), 1))], axis=1)
    projected = (m_to_px @ points.T).T
    pixels = np.asarray(projected[:, :2] / projected[:, 2:3])
    near_y = pixels[[0, 1, 2, 3, 4], 1].mean()
    far_y = pixels[[8, 9, 10, 11, 12], 1].mean()
    if near_y < far_y:  # "near" ended up at the top: flip the convention
        pixels = pixels[ROT180_IDX]
    return pixels


def yolo_pose_label(
    keypoints_px: np.ndarray, width: int, height: int, margin_px: float = 8.0
) -> str | None:
    """One-line YOLO-pose label for the court object, or None if too few points.

    The bounding box is the full frame. There is exactly one court per image,
    and a keypoint-enclosing box degenerates to a thin horizontal strip in
    court-level views (all visible points land near the horizon), which
    destabilizes box regression. A constant full-frame box removes that
    pathology; only the keypoints carry the geometry we actually use.
    """
    in_bounds = (
        (keypoints_px[:, 0] >= -margin_px)
        & (keypoints_px[:, 0] < width + margin_px)
        & (keypoints_px[:, 1] >= -margin_px)
        & (keypoints_px[:, 1] < height + margin_px)
    )
    if in_bounds.sum() < 4:
        return None
    parts = ["0 0.5 0.5 1.0 1.0"]
    for (x, y), ok in zip(keypoints_px, in_bounds, strict=True):
        if ok:
            nx = min(max(x / width, 0.0), 1.0)
            ny = min(max(y / height, 0.0), 1.0)
            parts.append(f"{nx:.6f} {ny:.6f} 2")
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


def propagate_static_court_labels(
    coco_json: Path,
    videos_dir: Path,
    output_dir: Path,
    frames_per_camera: int = 150,
    val_per_camera: int = 25,
    seed: int = 0,
) -> tuple[int, int]:
    """Stamp each static camera's court annotation onto many of its frames.

    PADELVIC cameras are on fixed tripods (verified: <1 px dispersion of the
    annotated points across frames), so the 13 court points sit at identical
    pixels in every frame of a video. We therefore replicate the single known
    annotation onto many frames sampled across the match: same court label,
    but with players in different positions and varying occlusion/lighting.
    This forces the detector to key off the court lines rather than the
    players, which 7 near-duplicate frames per view could not teach.

    Camera name is the annotation filename minus its frame-index suffix, and
    must match a `<camera>.mp4` under videos_dir. Returns (train, val) counts.
    """
    import random

    with open(coco_json) as f:
        coco = json.load(f)
    if coco["categories"][0]["keypoints"] != _court_keypoint_names():
        raise ValueError("keypoint order mismatch with the court schema")

    images_by_id = {i["id"]: i for i in coco["images"]}
    canonical: dict[str, np.ndarray] = {}
    for annotation in coco["annotations"]:
        camera = images_by_id[annotation["image_id"]]["file_name"].rsplit("_", 1)[0]
        canonical.setdefault(camera, np.array(annotation["keypoints"]).reshape(13, 3))

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}
    for camera, keypoints in sorted(canonical.items()):
        video_path = videos_dir / f"{camera}.mp4"
        if not video_path.exists():
            print(f"  aviso: falta el vídeo {video_path}, cámara {camera} omitida")
            continue
        # Points marked 'outside' in the annotation carry out-of-image coords;
        # yolo_pose_label flags those as not-visible automatically.
        keypoints_px = np.where(keypoints[:, 2:3] > 0, keypoints[:, :2], np.full((13, 2), -1e6))
        capture = cv2.VideoCapture(str(video_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        low, high = int(total * 0.05), int(total * 0.95)
        indices = sorted(rng.sample(range(low, high), min(frames_per_camera, high - low)))
        for position, frame_index in enumerate(indices):
            split = "val" if position >= len(indices) - val_per_camera else "train"
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            height, width = image.shape[:2]
            label = yolo_pose_label(keypoints_px, width, height)
            if label is None:
                continue
            images_out = output_dir / "images" / split
            labels_out = output_dir / "labels" / split
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            stem = f"{camera}_{frame_index:06d}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (labels_out / f"{stem}.txt").write_text(label + "\n")
            counts[split] += 1
        capture.release()
        print(f"  {camera}: {counts['train']} train / {counts['val']} val acumulados")
    print(f"propagadas: {counts['train']} train, {counts['val']} val -> {output_dir}")
    return counts["train"], counts["val"]


def _court_keypoint_names() -> list[str]:
    from padel_cv.court import COURT_KEYPOINT_NAMES

    return COURT_KEYPOINT_NAMES


def convert_coco_court_annotations(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    val_per_group: int = 3,
) -> tuple[int, int]:
    """Convert CVAT COCO-keypoints court annotations to YOLO-pose format.

    Images are grouped by camera (filename prefix); the last `val_per_group`
    of each group go to the val split so every camera is represented in both
    splits. Keypoints dragged outside the image (CVAT 'outside' convention)
    become visibility 0. Returns (train_count, val_count).
    """
    import shutil

    with open(coco_json) as f:
        coco = json.load(f)
    order = coco["categories"][0]["keypoints"]
    from padel_cv.court import COURT_KEYPOINT_NAMES

    if order != COURT_KEYPOINT_NAMES:
        raise ValueError(f"keypoint order mismatch: {order}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    by_group: dict[str, list[dict[str, Any]]] = {}
    for image in sorted(coco["images"], key=lambda i: i["file_name"]):
        by_group.setdefault(image["file_name"].rsplit("_", 1)[0], []).append(image)

    counts = {"train": 0, "val": 0}
    for group_images in by_group.values():
        for position, image in enumerate(group_images):
            split = "val" if position >= len(group_images) - val_per_group else "train"
            anns = annotations_by_image.get(image["id"], [])
            if len(anns) > 1:
                print(f"  aviso: {image['file_name']} tiene {len(anns)} skeletons; uso el primero")
            width, height = image["width"], image["height"]
            keypoints_px = np.zeros((13, 2))
            if anns:
                raw = anns[0]["keypoints"]
                for i in range(13):
                    x, y, v = raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]
                    # CVAT exports 'outside' points with out-of-image coords:
                    # push them far out so yolo_pose_label flags them v=0.
                    keypoints_px[i] = (x, y) if v > 0 else (-1e6, -1e6)
            label = yolo_pose_label(keypoints_px, width, height)
            images_out = output_dir / "images" / split
            labels_out = output_dir / "labels" / split
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(images_dir / image["file_name"], images_out / image["file_name"])
            stem = Path(image["file_name"]).stem
            (labels_out / f"{stem}.txt").write_text((label or "") + ("\n" if label else ""))
            counts[split] += 1
    print(f"convertidas: {counts['train']} train, {counts['val']} val -> {output_dir}")
    return counts["train"], counts["val"]

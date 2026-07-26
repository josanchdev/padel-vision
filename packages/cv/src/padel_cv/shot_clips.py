"""Build a shot-classification dataset from PadelTracker100 labels.

PadelTracker100 labels shots per frame (has_shot + type) and the ball per
frame, but never says *which* player hit. Attribution (ADR-0008): at the
impact frame — the frame in a shot run where the ball is closest to any
player's wrist — the owner of that nearest wrist is the hitter. Around that
frame we crop a fixed window of the hitter's normalized skeleton: one clip,
one label.

Pose source is a parameter: GT (ViTPose) validates the logic here; the real
training set uses our YOLO26 poses (ADR-0008), same code path.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

LEFT_WRIST, RIGHT_WRIST = 9, 10
MIN_KEYPOINT_CONFIDENCE = 0.3

# Level-2 taxonomy: Dropshot folds into "Other" (only 1-25 samples; ADR-0008).
SHOT_CLASSES = ["Forehand", "Backhand", "Smash", "Serve", "Other"]
_LABEL_MAP = {"Dropshot": "Other"}


def canonical_label(raw: str) -> str:
    return _LABEL_MAP.get(raw, raw)


@dataclass(frozen=True)
class ShotRun:
    """A contiguous run of has_shot=1 frames sharing one shot type."""

    start: int
    end: int
    shot_type: str

    @property
    def frames(self) -> range:
        return range(self.start, self.end + 1)


@dataclass(frozen=True)
class AttributedShot:
    run: ShotRun
    impact_frame: int
    player_index: int
    wrist_ball_dist_px: float


def _frame_index(file_name: str) -> int:
    return int(Path(file_name).stem.split("_")[1])


def load_ball_positions(ball_json: Path) -> dict[int, tuple[float, float]]:
    """Frame index -> ball center in pixels (category 1 = Ball)."""
    with open(ball_json) as f:
        coco = json.load(f)
    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in coco["images"]}
    positions: dict[int, tuple[float, float]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == 1:
            x, y, w, h = ann["bbox"]
            positions[frame_of[ann["image_id"]]] = (x + w / 2, y + h / 2)
    return positions


def load_shot_runs(shots_csv: Path) -> list[ShotRun]:
    """Contiguous has_shot=1 runs, each with its (single) shot type."""
    runs: list[ShotRun] = []
    start: int | None = None
    shot_type = ""
    with open(shots_csv) as f:
        for row in csv.DictReader(f, delimiter=";"):
            idx = _frame_index(row["file_name"])
            if row["has_shot"] == "1":
                if start is None:
                    start, shot_type = idx, row["category"]
            elif start is not None:
                runs.append(ShotRun(start, idx - 1, shot_type))
                start = None
    if start is not None:
        runs.append(ShotRun(start, start, shot_type))
    return runs


def load_gt_poses(pose_json: Path) -> dict[int, npt.NDArray[np.float32]]:
    """Frame index -> array (n_players, 17, 3) of GT skeletons."""
    with open(pose_json) as f:
        coco = json.load(f)
    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in coco["images"]}
    by_frame: dict[int, list[npt.NDArray[np.float32]]] = {}
    for ann in coco["annotations"]:
        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        kp[:, 2] = np.where(kp[:, 2] > 0, 1.0, 0.0)  # COCO visibility -> confidence
        by_frame.setdefault(frame_of[ann["image_id"]], []).append(kp)
    return {f: np.stack(skeletons) for f, skeletons in by_frame.items()}


def _wrist_ball_distance(skeleton: npt.NDArray[np.float32], ball: tuple[float, float]) -> float:
    """Smallest distance from either confident wrist to the ball (inf if none)."""
    best = float("inf")
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        x, y, conf = skeleton[wrist]
        if conf >= MIN_KEYPOINT_CONFIDENCE:
            best = min(best, float(np.hypot(x - ball[0], y - ball[1])))
    return best


def attribute_shot(
    run: ShotRun,
    ball_positions: dict[int, tuple[float, float]],
    poses: dict[int, npt.NDArray[np.float32]],
) -> AttributedShot | None:
    """Find the impact frame and hitting player for a shot run.

    Impact = the (frame, player) pair minimizing wrist-to-ball distance over
    the run. Returns None if no frame in the run has both a ball and poses.
    """
    best: AttributedShot | None = None
    for frame in run.frames:
        ball = ball_positions.get(frame)
        skeletons = poses.get(frame)
        if ball is None or skeletons is None:
            continue
        for player_index, skeleton in enumerate(skeletons):
            dist = _wrist_ball_distance(skeleton, ball)
            if best is None or dist < best.wrist_ball_dist_px:
                best = AttributedShot(run, frame, player_index, dist)
    return best


def attribute_all(
    ball_json: Path, shots_csv: Path, poses: dict[int, npt.NDArray[np.float32]]
) -> Iterator[AttributedShot]:
    ball_positions = load_ball_positions(ball_json)
    for run in load_shot_runs(shots_csv):
        shot = attribute_shot(run, ball_positions, poses)
        if shot is not None:
            yield shot

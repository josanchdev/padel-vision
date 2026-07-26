"""Ball dataset for TrackNet: COCO ball boxes -> Gaussian heatmap targets.

PadelTracker100 annotates the ball as a COCO bounding box (~8x8 px in 1080p).
TrackNet does not predict boxes; it predicts a **heatmap** where a 2D Gaussian
marks the ball centre. This module bridges the two: it reads the ball boxes,
takes each box centre, and renders a Gaussian target on a downscaled grid
(ADR-0009, Decision C).

The network input is a stack of 3 consecutive frames (the movement between them
is the signal that separates the tiny ball from the static background), so a
sample is (frames t-2, t-1, t) -> heatmap for frame t. Frames without a ball
annotation get an all-zero target (the ball is genuinely absent/occluded), which
teaches the network to output nothing rather than hallucinate.

Boxes come from the dataset only as a training kickstarter (ADR-0005): the
trained model detects the ball at inference from its own weights alone, never
from these annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

BALL_CATEGORY_ID = 1  # "Ball" in PadelTracker100 COCO categories

# TrackNet stacks this many consecutive frames as input to predict the last one.
INPUT_FRAMES = 3


@dataclass(frozen=True)
class BallFrame:
    """Ball centre for one frame, or None if unannotated.

    center_xy is FRACTIONAL (each coord in [0, 1], relative to frame width/
    height) so it is independent of the resolution frames are later cached at.
    """

    frame_index: int
    center_xy: tuple[float, float] | None
    occluded: bool


def load_ball_centers(ball_json: Path) -> dict[int, BallFrame]:
    """Read a PadelTracker100 *_ball.json into frame_index -> BallFrame.

    file_name is like ``frame_000123.PNG``; the numeric part is the frame index.
    Only the ``Ball`` category is kept (the file also holds wall/shot events).
    Centres are normalized to [0, 1] using each image's width/height.
    """
    data = json.loads(Path(ball_json).read_text())
    image_frame = {img["id"]: _frame_index(img["file_name"]) for img in data["images"]}
    image_wh = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    centers: dict[int, BallFrame] = {}
    for ann in data["annotations"]:
        if ann["category_id"] != BALL_CATEGORY_ID:
            continue
        frame = image_frame[ann["image_id"]]
        img_w, img_h = image_wh[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        occluded = bool(ann.get("attributes", {}).get("occluded", False))
        center = ((x + w / 2.0) / img_w, (y + h / 2.0) / img_h)
        centers[frame] = BallFrame(frame, center, occluded)
    return centers


def _frame_index(file_name: str) -> int:
    stem = Path(file_name).stem  # frame_000123
    return int(stem.split("_")[-1])


def render_heatmap(
    center_xy: tuple[float, float] | None,
    grid_wh: tuple[int, int],
    sigma: float = 2.5,
) -> FloatArray:
    """Render a Gaussian blob at the ball centre on a grid of size grid_wh.

    center_xy is FRACTIONAL ([0, 1] per axis); it is scaled onto the grid.
    None -> all-zero heatmap (ball absent). Fully vectorized.
    """
    grid_w, grid_h = grid_wh
    target = np.zeros((grid_h, grid_w), dtype=np.float32)
    if center_xy is None:
        return target

    cx = center_xy[0] * grid_w
    cy = center_xy[1] * grid_h

    ys = np.arange(grid_h, dtype=np.float32)[:, None]
    xs = np.arange(grid_w, dtype=np.float32)[None, :]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist_sq / (2.0 * sigma**2)).astype(np.float32)

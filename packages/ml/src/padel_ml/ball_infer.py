"""Run a trained ball detector over a video — for validation and pre-labeling.

Two uses (Phase 1d / Phase 2 of the ball work):
- **Validation**: write an mp4 with the ball marked, to eyeball how well the
  model does on a new court/lighting.
- **Pre-labeling**: export per-frame ball proposals as CVAT annotations, so
  labelling new footage means correcting proposals, not marking from scratch.

Works purely in image space: no court/homography needed (the ball is detected in
pixels), so it runs on any video regardless of court type. This is deliberately
standalone — not the full pose+court+shots pipeline.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from padel_cv.ball_cache import FRAME_H, FRAME_W
from padel_cv.ball_data import INPUT_FRAMES
from padel_ml.ball_metrics import peak_xy
from padel_ml.ball_stage import _build_model


@dataclass
class BallHit:
    """A ball detection in one frame (image pixels)."""

    frame_index: int
    x_px: float
    y_px: float
    confidence: float


class BallDetector:
    """Stateful per-frame ball detector (buffers INPUT_FRAMES, no homography)."""

    def __init__(
        self, checkpoint: Path, min_confidence: float = 0.5, device: str | None = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self._model = _build_model(ckpt.get("model_name", "tracknetv2"))
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(device).eval()
        self._min_confidence = min_confidence
        self._buffer: deque[np.ndarray] = deque(maxlen=INPUT_FRAMES)

    def detect(self, image: np.ndarray, frame_index: int) -> BallHit | None:
        """Feed one BGR frame; return a BallHit once the buffer is full."""
        small = cv2.resize(image, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        self._buffer.append(np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1)))
        if len(self._buffer) < INPUT_FRAMES:
            return None
        stacked = np.concatenate(list(self._buffer), axis=0)  # (9, H, W)
        with torch.no_grad():
            batch = torch.from_numpy(stacked)[None].to(self._device)
            heatmap = self._model(batch)[0, 0].cpu()
        confidence = float(heatmap.max())
        if confidence < self._min_confidence:
            return None
        gx, gy = peak_xy(heatmap)
        grid_h, grid_w = heatmap.shape
        img_h, img_w = image.shape[:2]
        return BallHit(
            frame_index=frame_index,
            x_px=(gx + 0.5) / grid_w * img_w,
            y_px=(gy + 0.5) / grid_h * img_h,
            confidence=confidence,
        )


def detect_ball_in_video(
    video_path: Path,
    checkpoint: Path,
    min_confidence: float = 0.5,
    max_frames: int | None = None,
    on_frame: Callable[[int, np.ndarray, BallHit | None], None] | None = None,
) -> list[BallHit]:
    """Run the detector over a video, returning every ball hit.

    on_frame, if given, is called as on_frame(frame_index, bgr_image, hit_or_None)
    per frame — used by the video-writer overlay without a second decode pass.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    detector = BallDetector(checkpoint, min_confidence)
    hits: list[BallHit] = []
    index = 0
    while True:
        if max_frames is not None and index >= max_frames:
            break
        ok, image = capture.read()
        if not ok:
            break
        hit = detector.detect(image, index)
        if hit is not None:
            hits.append(hit)
        if on_frame is not None:
            on_frame(index, image, hit)
        index += 1
    capture.release()
    return hits

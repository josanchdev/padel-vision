"""BallDetectionStage: run TrackNet over the pipeline and locate the ball.

TrackNet needs 3 consecutive frames, but the pipeline hands stages one frame at
a time, so this stage buffers the last INPUT_FRAMES frames internally (stages
keep their own temporal state — see pipeline docs). Once the buffer is full it
runs the model, takes the heatmap peak, and if it clears a confidence threshold
records a BallDetection: the peak in full-resolution pixels plus, when a
homography is present, the point projected to court metres.

The model is *our own* trained TrackNet; nothing here reads external ball
annotations (ADR-0009/ADR-0005). Lives in packages/ml so packages/cv keeps no
hard torch dependency; the CLI imports it lazily.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from padel_cv.ball_cache import FRAME_H, FRAME_W
from padel_cv.ball_data import INPUT_FRAMES
from padel_cv.court import project_point
from padel_cv.pipeline import BallDetection, Frame
from padel_ml.ball_metrics import peak_xy
from padel_ml.tracknet import TrackNetV2
from padel_ml.tracknet_v3 import TrackNetV3


def _build_model(model_name: str) -> torch.nn.Module:
    if model_name == "tracknetv2":
        return TrackNetV2()
    if model_name == "tracknetv3":
        return TrackNetV3()
    raise ValueError(f"unknown ball model: {model_name}")


class BallDetectionStage:
    """Detect the ball per frame with a trained TrackNet checkpoint."""

    def __init__(self, checkpoint: Path, min_confidence: float = 0.5) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._model = _build_model(ckpt.get("model_name", "tracknetv2"))
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._min_confidence = min_confidence
        # Rolling buffer of the last INPUT_FRAMES downscaled frames (channels-
        # first float [0,1]); a detection needs a full window.
        self._buffer: deque[np.ndarray] = deque(maxlen=INPUT_FRAMES)

    def process(self, frame: Frame) -> Frame:
        small = cv2.resize(frame.image, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        chw = np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1))  # (3, H, W)
        self._buffer.append(chw)
        if len(self._buffer) < INPUT_FRAMES:
            return frame  # not enough context yet

        stacked = np.concatenate(list(self._buffer), axis=0)  # (9, H, W)
        with torch.no_grad():
            heatmap = self._model(torch.from_numpy(stacked)[None])[0, 0]  # (gh, gw)
        confidence = float(heatmap.max())
        if confidence < self._min_confidence:
            return frame

        gx, gy = peak_xy(heatmap)
        grid_h, grid_w = heatmap.shape
        # Peak -> fractional -> full-resolution pixels of THIS frame.
        img_h, img_w = frame.image.shape[:2]
        x_px = (gx + 0.5) / grid_w * img_w
        y_px = (gy + 0.5) / grid_h * img_h

        court_xy: tuple[float, float] | None = None
        if frame.homography is not None:
            court_xy = project_point(frame.homography, x_px, y_px)
        frame.ball = BallDetection(
            image_xy=(x_px, y_px), confidence=confidence, court_xy_m=court_xy
        )
        return frame

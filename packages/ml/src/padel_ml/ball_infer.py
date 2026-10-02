"""Run a trained TrackNet over frames: the ball position in each one.

Works purely in image space: no court/homography needed (the ball is detected in
pixels). `rally_analysis` feeds it every frame of a rally and cleans the
trajectory afterwards (`ball_postprocess`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from padel_cv.ball_data import INPUT_FRAMES
from padel_ml.ball_metrics import peak_xy
from padel_ml.tracknet import TrackNetV2
from padel_ml.tracknet_v3 import TrackNetV3


def build_ball_model(model_name: str) -> torch.nn.Module:
    """An untrained TrackNet by name, as stored in the checkpoints."""
    if model_name == "tracknetv2":
        return TrackNetV2()
    if model_name == "tracknetv3":
        return TrackNetV3()
    raise ValueError(f"unknown ball model: {model_name}")


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
        self,
        checkpoint: Path,
        min_confidence: float = 0.5,
        device: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self._model = build_ball_model(ckpt.get("model_name", "tracknetv2"))
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(device).eval()
        self._min_confidence = min_confidence
        # The network is fully convolutional, so it accepts a larger frame than
        # the 512x288 it trained on. Measured over a full rally, 768x432 finds
        # the ball in 91.0% of frames against 89.4%, and — the point — cuts
        # frozen detections from 14.3% to 8.9%: a bigger ball is harder to
        # confuse with a static distractor. 1024x576 goes too far from the
        # training domain and detection drops back to 87.9%.
        self._frame_size = frame_size or (768, 432)
        self._buffer: deque[np.ndarray] = deque(maxlen=INPUT_FRAMES)

    def reset(self) -> None:
        """Forget the buffered frames before starting a new video.

        The network sees INPUT_FRAMES consecutive frames at once, so without this
        the first detections of a video would be computed partly from the tail
        of the previous one.
        """
        self._buffer.clear()

    def detect(self, image: np.ndarray, frame_index: int) -> BallHit | None:
        """Feed one BGR frame; return a BallHit once the buffer is full."""
        small = cv2.resize(image, self._frame_size, interpolation=cv2.INTER_AREA)
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

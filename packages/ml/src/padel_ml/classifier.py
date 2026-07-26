"""Run the trained PoseConv3D on a skeleton window at inference time.

Loads an archived checkpoint (state dict + class names) and classifies a
(T, 17, 3) clip of one player's normalized skeleton. This is the "what" half of
shot recognition; the "when" is proposed by the wrist-speed heuristic (our own
signal, no external annotation) and filtered here via the NoShot class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from padel_cv.clip_builder import NO_SHOT_LABEL, normalize_skeleton
from padel_ml.heatmap import batch_to_heatmaps
from padel_ml.poseconv3d import PoseConv3D

FloatArray = npt.NDArray[np.float32]


class ShotClassifier:
    """Classifies a normalized skeleton window into a shot type or NoShot."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = list(ckpt["classes"])
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = PoseConv3D(num_classes=len(self.classes)).to(self._device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

    def classify_window(self, raw_window: FloatArray) -> tuple[str, float]:
        """Normalize a (T, 17, 3) window, classify it. Returns (label, prob)."""
        normalized = np.stack([normalize_skeleton(frame) for frame in raw_window])
        clip = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,T,17)
        with torch.no_grad():
            volume = batch_to_heatmaps(clip).to(self._device)
            probs = self._model(volume).softmax(dim=1)[0].cpu().numpy()
        best = int(probs.argmax())
        return self.classes[best], float(probs[best])

    def is_shot(self, label: str) -> bool:
        return label != NO_SHOT_LABEL

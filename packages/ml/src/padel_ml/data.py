"""Load the persisted shot-clip dataset for training.

Clips are stored as (N, T, 17, 3); models want (N, 3, T, 17). We split by match
(train on one final, validate on the other) so the model is tested on unseen
players — a harder, more honest measure than a random split (ADR-0008).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset


@dataclass
class ShotData:
    train: TensorDataset
    val: TensorDataset
    classes: list[str]
    class_weights: torch.Tensor  # inverse-frequency, for the loss


def _to_model_layout(keypoints: npt.NDArray[np.float32]) -> torch.Tensor:
    # (N, T, V, C) -> (N, C, T, V)
    return torch.from_numpy(keypoints).permute(0, 3, 1, 2).contiguous().float()


def load_shot_data(npz_path: Path, val_match: int = 0) -> ShotData:
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["keypoints"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    matches = data["matches"].astype(np.int64)
    classes = [str(c) for c in data["classes"]]

    is_val = matches == val_match
    x_train = _to_model_layout(keypoints[~is_val])
    x_val = _to_model_layout(keypoints[is_val])
    y_train = torch.from_numpy(labels[~is_val])
    y_val = torch.from_numpy(labels[is_val])

    # Inverse-frequency class weights from the training split (ADR-0008).
    counts = np.bincount(labels[~is_val], minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / (len(classes) * np.maximum(counts, 1.0))
    return ShotData(
        train=TensorDataset(x_train, y_train),
        val=TensorDataset(x_val, y_val),
        classes=classes,
        class_weights=torch.from_numpy(weights.astype(np.float32)),
    )

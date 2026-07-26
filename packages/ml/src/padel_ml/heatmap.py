"""Turn skeleton clips into heatmap volumes for PoseConv3D.

Each joint becomes a 2D Gaussian blob on a small HxW grid, one channel per
joint, stacked over time: a clip (C=3, T, V=17) -> volume (V=17, T, H, W).
Unlike a graph, a 3D CNN over these volumes is robust to noisy keypoints
(our YOLO poses) because a blurry blob degrades gracefully.

Normalized skeletons are hip-centered/torso-scaled (roughly [-3, 3]); we map
that range onto the grid. Low-confidence joints get a dimmer blob.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float32]

COORD_RANGE = 3.0  # normalized coords mostly fall within +-3 torso lengths


def clip_to_heatmap(clip: FloatArray, size: int = 32, sigma: float = 1.5) -> torch.Tensor:
    """(C=3, T, V=17) normalized skeleton -> (V, T, size, size) heatmap volume.

    Fully vectorized (no per-frame/joint loops): each joint's (x, y) is turned
    into a Gaussian over the grid via broadcasting, weighted by confidence.
    """
    grid = np.linspace(-COORD_RANGE, COORD_RANGE, size, dtype=np.float32)
    cell = (2 * COORD_RANGE) / size  # normalized units per grid cell
    two_sigma_sq = 2.0 * sigma**2

    xs = clip[0].T[:, :, None, None]  # (V, T, 1, 1)
    ys = clip[1].T[:, :, None, None]
    conf = clip[2].T[:, :, None, None]
    gx = grid[None, None, None, :]  # (1, 1, 1, size)
    gy = grid[None, None, :, None]  # (1, 1, size, 1)

    dist_sq = ((gx - xs) ** 2 + (gy - ys) ** 2) / (cell**2)
    volume = np.exp(-dist_sq / two_sigma_sq) * np.clip(conf, 0.0, None)
    return torch.from_numpy(volume.astype(np.float32))


def batch_to_heatmaps(clips: torch.Tensor, size: int = 32, sigma: float = 1.5) -> torch.Tensor:
    """(N, C, T, V) -> (N, V, T, size, size)."""
    volumes = [clip_to_heatmap(clips[i].numpy(), size, sigma) for i in range(clips.shape[0])]
    return torch.stack(volumes)

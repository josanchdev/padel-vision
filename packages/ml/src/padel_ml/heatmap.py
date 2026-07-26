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
    """(C=3, T, V=17) normalized skeleton -> (V, T, size, size) heatmap volume."""
    _, num_frames, num_joints = clip.shape
    grid = np.linspace(-COORD_RANGE, COORD_RANGE, size, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)  # (size, size)
    volume = np.zeros((num_joints, num_frames, size, size), dtype=np.float32)
    two_sigma_sq = 2.0 * sigma**2
    cell = (2 * COORD_RANGE) / size  # normalized units per grid cell
    for t in range(num_frames):
        for j in range(num_joints):
            x, y, conf = clip[:, t, j]
            if conf <= 0.0:
                continue
            # Distance in grid cells so sigma is expressed in cells.
            dist_sq = ((gx - x) ** 2 + (gy - y) ** 2) / (cell**2)
            volume[j, t] = np.exp(-dist_sq / two_sigma_sq) * float(conf)
    return torch.from_numpy(volume)


def batch_to_heatmaps(clips: torch.Tensor, size: int = 32, sigma: float = 1.5) -> torch.Tensor:
    """(N, C, T, V) -> (N, V, T, size, size)."""
    volumes = [clip_to_heatmap(clips[i].numpy(), size, sigma) for i in range(clips.shape[0])]
    return torch.stack(volumes)

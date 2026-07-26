"""Skeleton clip augmentation for the small shot dataset.

With ~1.3k clips, augmentation matters as much as architecture. Transforms act
on normalized clips (C=3: x, y, confidence) already hip-centered/torso-scaled:

- horizontal flip: negate x AND swap left/right joint indices (a mirrored
  skeleton with unswapped indices is anatomically broken) — doubles the
  effective left/right-handed variety.
- small rotation / scale: camera/viewpoint jitter the normalization leaves.
- temporal jitter: shift the window a few frames so the impact is not always
  dead-center.

Confidence channel is never rotated/scaled — only the (x, y) rows.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# COCO-17 left<->right index swaps for a horizontal flip.
_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

FloatArray = npt.NDArray[np.float32]


def flip_horizontal(clip: FloatArray) -> FloatArray:
    """Mirror x and swap left/right joints. clip: (C=3, T, V=17)."""
    out = clip.copy()
    out[0] = -out[0]  # negate x
    perm = np.arange(out.shape[2])
    for left, right in _FLIP_PAIRS:
        perm[left], perm[right] = right, left
    return out[:, :, perm]


def rotate_scale(
    clip: FloatArray, max_deg: float, max_scale: float, rng: np.random.Generator
) -> FloatArray:
    """Apply one random small rotation + isotropic scale to (x, y)."""
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    scale = rng.uniform(1 - max_scale, 1 + max_scale)
    cos, sin = np.cos(theta) * scale, np.sin(theta) * scale
    out = clip.copy()
    x, y = out[0], out[1]
    out[0] = cos * x - sin * y
    out[1] = sin * x + cos * y
    return out


def augment_clip(
    clip: FloatArray,
    rng: np.random.Generator,
    flip_prob: float = 0.5,
    max_deg: float = 12.0,
    max_scale: float = 0.1,
) -> FloatArray:
    """Random augmentation pipeline for one training clip."""
    out = clip
    if rng.random() < flip_prob:
        out = flip_horizontal(out)
    out = rotate_scale(out, max_deg, max_scale, rng)
    return out

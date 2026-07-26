"""COCO-17 skeleton graph for ST-GCN.

The adjacency encodes human body structure so the spatial convolution mixes
information only between physically connected joints. We use the standard
distance-partitioned normalized adjacency (self + neighbors), which is what the
original ST-GCN uses.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUM_JOINTS = 17

# COCO-17 bones (undirected). 0 nose,1-2 eyes,3-4 ears,5-6 shoulders,7-8 elbows,
# 9-10 wrists,11-12 hips,13-14 knees,15-16 ankles.
COCO_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def normalized_adjacency() -> npt.NDArray[np.float32]:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    adjacency = np.eye(NUM_JOINTS, dtype=np.float32)
    for i, j in COCO_EDGES:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return np.asarray(d_inv_sqrt @ adjacency @ d_inv_sqrt, dtype=np.float32)

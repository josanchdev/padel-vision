"""PyTorch Dataset over the ball frame cache for TrackNet training.

A TrackNet sample is (frames t-2, t-1, t) -> ball heatmap for frame t. The three
input frames must be *consecutive in the original video*: stacking across a gap
(a dropped frame, or the boundary between two matches) would feed the network a
fake motion cue. So we build a flat frame table from the cache and only emit
windows whose INPUT_FRAMES indices are strictly consecutive.

Frames are stored downscaled uint8 BGR; here they become float [0,1], channels
-first, and the three frames are concatenated on the channel axis (9 channels).
Loading is lazy: windows share frames, so we keep the per-match frame arrays in
memory once (a match at 512x288 is a few GB) and slice windows on __getitem__.

The heatmap grid is smaller than the frame (a training-time choice); centres are
stored in original pixels, so render_heatmap rescales them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from padel_cv.ball_cache import FRAME_H, FRAME_W
from padel_cv.ball_data import INPUT_FRAMES, render_heatmap

FloatArray = npt.NDArray[np.float32]

# Heatmap output grid. Quarter of the input resolution keeps the ball blob a few
# cells wide while making the target cheap to learn (TrackNetV2 uses this ratio).
GRID_W = FRAME_W // 4  # 128
GRID_H = FRAME_H // 4  # 72


@dataclass
class _MatchFrames:
    """One match's cached frames, in ascending frame order."""

    frames: npt.NDArray[np.uint8]  # (N, H, W, 3) BGR
    indices: npt.NDArray[np.int64]  # (N,) original frame index
    centers: FloatArray  # (N, 2) fractional ball centre, NaN if absent
    occluded: npt.NDArray[np.bool_]  # (N,) ball marked occluded


def _load_match(cache_dir: Path) -> _MatchFrames:
    """Concatenate all shards of one match into ascending-order arrays."""
    shards = sorted(cache_dir.glob("frames_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No frame shards in {cache_dir}")
    frames, indices, centers, occluded = [], [], [], []
    for shard in shards:
        data = np.load(shard)
        frames.append(data["frames"])
        indices.append(data["indices"])
        centers.append(data["centers"])
        occluded.append(data["occluded"])
    idx = np.concatenate(indices)
    order = np.argsort(idx)
    return _MatchFrames(
        frames=np.concatenate(frames)[order],
        indices=idx[order],
        centers=np.concatenate(centers)[order],
        occluded=np.concatenate(occluded)[order],
    )


def _consecutive_windows(indices: npt.NDArray[np.int64]) -> list[int]:
    """Positions p such that indices[p-2:p+1] are strictly consecutive frames.

    Returns the position of the LAST frame of each valid window (the one the
    heatmap targets), so a window is frames [p-INPUT_FRAMES+1 .. p].
    """
    ends: list[int] = []
    span = INPUT_FRAMES - 1
    for p in range(span, len(indices)):
        if indices[p] - indices[p - span] == span:
            ends.append(p)
    return ends


class BallClips(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Windows of INPUT_FRAMES consecutive frames -> ball heatmap for the last.

    Each item is (stacked_frames, heatmap, occluded_flag); the flag drives the
    visible-vs-occluded metric split (ADR-0009).
    """

    def __init__(self, cache_dirs: list[Path], sigma: float = 2.5) -> None:
        self._sigma = sigma
        self._matches: list[_MatchFrames] = []
        # Flat index of (match, end_position) for every valid window.
        self._windows: list[tuple[int, int]] = []
        for cache_dir in cache_dirs:
            match = _load_match(cache_dir)
            m = len(self._matches)
            self._matches.append(match)
            self._windows.extend((m, p) for p in _consecutive_windows(match.indices))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, end = self._windows[index]
        match = self._matches[m]
        start = end - (INPUT_FRAMES - 1)
        # (INPUT_FRAMES, H, W, 3) BGR uint8 -> concat on channels -> (9, H, W).
        window = match.frames[start : end + 1].astype(np.float32) / 255.0
        stacked = np.transpose(window, (0, 3, 1, 2)).reshape(-1, FRAME_H, FRAME_W)

        center = match.centers[end]
        center_xy = None if np.isnan(center).any() else (float(center[0]), float(center[1]))
        heatmap = render_heatmap(center_xy, grid_wh=(GRID_W, GRID_H), sigma=self._sigma)
        occluded = torch.tensor(bool(match.occluded[end]))
        return torch.from_numpy(stacked), torch.from_numpy(heatmap), occluded

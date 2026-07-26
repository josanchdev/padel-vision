"""Cache downscaled video frames + ball centres for TrackNet training.

TrackNet takes 3 consecutive downscaled frames and predicts a ball heatmap for
the last one. Building that dataset means pairing each frame of a match with its
ball centre (from the PadelTracker100 *_ball.json, used only as a training
kickstarter — ADR-0009/ADR-0005). We cache frames at TrackNet's working
resolution (512x288) so training does not re-decode the 1080p video every epoch.

Crash-safe like the pose cache: frames are written in shards of `shard_size`,
a re-run skips shards already on disk. Storing frames once (not every 3-frame
window) avoids 3x duplication — the Dataset stacks windows lazily at load time.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from padel_cv.ball_data import load_ball_centers

# TrackNet's standard working resolution (16:9, keeps the court aspect).
FRAME_W = 512
FRAME_H = 288
SHARD_SIZE = 1000

UInt8Array = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]


def _shard_path(cache_dir: Path, start: int) -> Path:
    return cache_dir / f"frames_{start:07d}.npz"


def _covered_frames(cache_dir: Path) -> int:
    """Length of the contiguous frame prefix [0, N) already cached."""
    if not cache_dir.exists():
        return 0
    present: set[int] = set()
    for shard in cache_dir.glob("frames_*.npz"):
        present.update(int(f) for f in np.load(shard)["indices"])
    covered = 0
    while covered in present:
        covered += 1
    return covered


def extract_ball_frames_to_cache(
    video_path: Path,
    ball_json: Path,
    cache_dir: Path,
    shard_size: int = SHARD_SIZE,
    max_frames: int | None = None,
) -> int:
    """Cache downscaled frames + per-frame ball centres, resumably.

    Each shard stores: downscaled BGR frames (N, H, W, 3) uint8, their frame
    indices, and ball centres in ORIGINAL pixels (N, 2) float32 with NaN where
    the ball is unannotated. Centres stay in original pixels so the heatmap grid
    size is a training-time choice, not baked into the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    centers = load_ball_centers(ball_json)

    start_frame = _covered_frames(cache_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    written = 0
    shard_frames: list[UInt8Array] = []
    shard_indices: list[int] = []
    shard_centers: list[tuple[float, float]] = []

    def flush(start: int) -> None:
        if not shard_frames:
            return
        np.savez_compressed(
            _shard_path(cache_dir, start),
            frames=np.stack(shard_frames),
            indices=np.asarray(shard_indices, dtype=np.int64),
            centers=np.asarray(shard_centers, dtype=np.float32),
        )

    shard_start = start_frame
    while True:
        if max_frames is not None and written >= max_frames:
            break
        ok, image = capture.read()
        if not ok:
            break
        small = cv2.resize(image, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        ball = centers.get(frame_index)
        center = ball.center_xy if ball is not None and ball.center_xy is not None else None
        shard_frames.append(small.astype(np.uint8))
        shard_indices.append(frame_index)
        shard_centers.append(center if center is not None else (np.nan, np.nan))

        frame_index += 1
        written += 1
        if len(shard_frames) >= shard_size:
            flush(shard_start)
            shard_frames, shard_indices, shard_centers = [], [], []
            shard_start = frame_index
    flush(shard_start)
    capture.release()
    return written

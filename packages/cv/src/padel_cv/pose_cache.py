"""Cache YOLO26-pose + tracking results per video, for offline dataset building.

Running the pose detector over a full match is slow, so we do it once and cache
the per-frame skeletons (with track ids) to disk. These are *our own* poses —
the shot classifier trains on them so there is no gap between training and
inference, and nothing at runtime depends on external annotations (see
docs/decisions/0008 and the dataset-independence principle).

Crash-safe: results are written in shards of `shard_size` frames. A re-run
skips shards already on disk and resumes from the next frame, which matters on
this WSL box where long GPU jobs can crash. Track ids reset at shard
boundaries, but a shot window (~17 frames) is far shorter than a shard, so
within-window continuity always holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from padel_cv.pipeline import Pipeline
from padel_cv.stages import PlayerPoseStage

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass
class FramePoses:
    """All detections in one frame: parallel arrays indexed by detection."""

    keypoints: FloatArray  # (n, 17, 3)
    track_ids: IntArray  # (n,) -1 when the tracker gave no id
    boxes: FloatArray  # (n, 4) xyxy


def _shard_path(cache_dir: Path, start: int) -> Path:
    return cache_dir / f"shard_{start:07d}.npz"


def _covered_frames(cache_dir: Path, shard_size: int) -> int:
    """Length of the contiguous frame prefix [0, N) already on disk.

    Reads each shard's actual frame indices rather than assuming a full
    shard_size, so a partial final shard (e.g. from a --max-frames run or a
    clean early finish) resumes correctly.
    """
    if not cache_dir.exists():
        return 0
    present: set[int] = set()
    for shard in cache_dir.glob("shard_*.npz"):
        present.update(int(f) for f in np.load(shard)["frames"])
    covered = 0
    while covered in present:
        covered += 1
    return covered


def extract_poses_to_cache(
    video_path: Path,
    cache_dir: Path,
    confidence: float = 0.3,
    image_size: int = 1920,
    shard_size: int = 5000,
    max_frames: int | None = None,
) -> int:
    """Detect and track poses across a video, writing resumable shards.

    Returns the number of frames processed in this run (0 if already complete).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume_from = _covered_frames(cache_dir, shard_size)

    pipeline = Pipeline([PlayerPoseStage(confidence=confidence, image_size=image_size)])
    shard_frames: list[int] = []
    shard_kpts: list[FloatArray] = []
    shard_tracks: list[IntArray] = []
    shard_boxes: list[FloatArray] = []
    shard_counts: list[int] = []
    shard_start = resume_from
    processed = 0

    def flush(next_start: int) -> None:
        if not shard_counts:
            return
        keypoints = np.concatenate(shard_kpts) if shard_kpts else np.zeros((0, 17, 3), np.float32)
        track_ids = np.concatenate(shard_tracks) if shard_tracks else np.zeros(0, np.int64)
        boxes = np.concatenate(shard_boxes) if shard_boxes else np.zeros((0, 4), np.float32)
        np.savez_compressed(
            _shard_path(cache_dir, shard_start),
            frames=np.array(shard_frames, dtype=np.int64),
            counts=np.array(shard_counts, dtype=np.int64),
            keypoints=keypoints,
            track_ids=track_ids,
            boxes=boxes,
        )
        shard_frames.clear()
        shard_kpts.clear()
        shard_tracks.clear()
        shard_boxes.clear()
        shard_counts.clear()

    for frame in pipeline.run(str(video_path), start_frame=resume_from):
        if max_frames is not None and processed >= max_frames:
            break
        if frame.index > 0 and frame.index % shard_size == 0 and frame.index > shard_start:
            flush(frame.index)
            shard_start = frame.index
        n = len(frame.poses)
        shard_frames.append(frame.index)
        shard_counts.append(n)
        if n:
            ids = [p.track_id if p.track_id is not None else -1 for p in frame.poses]
            shard_kpts.append(np.stack([p.keypoints for p in frame.poses]))
            shard_tracks.append(np.array(ids, np.int64))
            shard_boxes.append(np.array([p.bbox_xyxy for p in frame.poses], np.float32))
        processed += 1
        if processed % 1000 == 0:
            print(f"  {frame.index} frames ({processed} this run)")
    flush(-1)
    print(f"{video_path.name}: {processed} frames procesados (desde {resume_from})")
    return processed


class PoseCache:
    """Read-only view over the cached shards of one video."""

    def __init__(self, cache_dir: Path) -> None:
        self._frames: dict[int, FramePoses] = {}
        for shard in sorted(cache_dir.glob("shard_*.npz")):
            data = np.load(shard)
            keypoints, track_ids, boxes = data["keypoints"], data["track_ids"], data["boxes"]
            offset = 0
            for frame_index, count in zip(data["frames"], data["counts"], strict=True):
                self._frames[int(frame_index)] = FramePoses(
                    keypoints=keypoints[offset : offset + count],
                    track_ids=track_ids[offset : offset + count],
                    boxes=boxes[offset : offset + count],
                )
                offset += count

    def __len__(self) -> int:
        return len(self._frames)

    def get(self, frame_index: int) -> FramePoses | None:
        return self._frames.get(frame_index)

    def keypoints_by_frame(self) -> dict[int, FloatArray]:
        """Frame -> (n, 17, 3), the format the attribution code expects."""
        return {f: fp.keypoints for f, fp in self._frames.items()}

"""Build training windows for the shot-type classifier (ADR-0016 F).

One window per labelled hit, holding what BST feeds its model: the hitter's pose
sequence and the ball trajectory over that window.

The window is ADAPTIVE, not fixed-width — this is the part of BST that measured
best (Min-F1 0.5210 -> 0.5822 on their hardest class). It runs from the
opponent's previous hit to their next hit plus a few frames, so the clip carries
the whole stroke (wind-up, contact, follow-through) *and* the beginning of how
the opponent replies, which lets the model infer the stroke type backwards. A
fixed width faces a dilemma instead: too short cuts the gesture, too long drags
in someone else's shot.

Sequences are resampled to a fixed length so they can be batched, and pose is
normalised per-window (hip-centred, torso-scaled) so the model sees gestures
rather than where on the court the player happened to stand.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from padel_ml.hit_assignment import FrameState

FloatArray = npt.NDArray[np.float32]

CLASSES = ["Forehand", "Backhand", "Smash", "Serve"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
DISCARD = "Other"

SEQ_LEN = 100
"""Frames per window after resampling (BST uses 100 with its adaptive strategy)."""

MAX_HALF_SPAN_S = 1.5
"""Paper's cap: a window never reaches further than this from the target hit."""

EPSILON_RATIO = 0.5
"""Extra tail past the opponent's next hit, as a fraction of `t` (BST: eps = t/2)."""

DEFAULT_T_S = 0.5
"""Half-window when there is no neighbouring hit to key off (BST: half the fps)."""

LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


@dataclass
class ShotWindow:
    """One labelled hit, ready for the model."""

    pose: FloatArray
    """(SEQ_LEN, 17, 3) — hitter's normalised keypoints + confidence."""
    ball: FloatArray
    """(SEQ_LEN, 3) — ball x, y (pose-normalised) and a present/missing flag."""
    label: int
    rally: str
    tournament: str
    hit_frame: int


def load_labels(csv_path: Path) -> list[tuple[int, str]]:
    """(frame, type) for every labelled hit of a rally, ordered by frame."""
    out: list[tuple[int, str]] = []
    for row in csv.DictReader(csv_path.open(), delimiter=";"):
        if row["type"]:
            out.append((int(row["frame"]), row["type"]))
    return sorted(out)


def window_bounds(hit_frames: list[int], index: int, fps: float) -> tuple[int, int]:
    """Adaptive window for hit `index`, following BST §3.1.

    Runs from the previous hit to the next hit plus eps. Both ends fall back to
    a fixed half-window when the hit is the first or last of the rally, and both
    are clamped so the window never stretches further than MAX_HALF_SPAN_S from
    the target — a long gap between points must not drag unrelated footage in.
    """
    target = hit_frames[index]
    t_frames = DEFAULT_T_S * fps
    cap = MAX_HALF_SPAN_S * fps

    start = max(hit_frames[index - 1], target - cap) if index > 0 else target - t_frames
    end = (
        min(hit_frames[index + 1] + EPSILON_RATIO * t_frames, target + cap)
        if index < len(hit_frames) - 1
        else target + t_frames
    )
    return round(start), round(end)


def _resample_indices(start: int, end: int, length: int) -> npt.NDArray[np.int64]:
    """`length` frame indices spanning [start, end] (repeats when the span is short)."""
    if end <= start:
        return np.full(length, start, dtype=np.int64)
    return np.round(np.linspace(start, end, length)).astype(np.int64)


def _hip_centre_and_scale(keypoints: FloatArray) -> tuple[FloatArray, float]:
    """Origin and size of a skeleton: mid-hip, and hip-to-shoulder distance.

    Scaling by torso length makes a near player and a far player produce the
    same numbers for the same gesture, which is what the classifier must learn.
    """
    hips = keypoints[[LEFT_HIP, RIGHT_HIP], :2].mean(axis=0)
    shoulders = keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER], :2].mean(axis=0)
    scale = float(np.hypot(*(shoulders - hips)))
    return hips.astype(np.float32), max(scale, 1.0)


def normalise_pose(keypoints: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Hip-centred, torso-scaled keypoints; also returns the centre and scale."""
    centre, scale = _hip_centre_and_scale(keypoints)
    out = keypoints.copy()
    out[:, :2] = (out[:, :2] - centre) / scale
    return out, centre, scale


def _hitter_at(poses: list[FloatArray], player_ids: list[int], hitter_id: int) -> FloatArray | None:
    for keypoints, player_id in zip(poses, player_ids, strict=True):
        if player_id == hitter_id:
            return keypoints
    return None


def build_window(
    keypoints_by_frame: dict[int, list[FloatArray]],
    player_ids_by_frame: dict[int, list[int]],
    ball_by_frame: dict[int, tuple[float, float]],
    hit_frames: list[int],
    index: int,
    hitter_id: int,
    fps: float,
    label: int,
    rally: str,
    tournament: str,
    seq_len: int = SEQ_LEN,
) -> ShotWindow | None:
    """Cut and normalise one window, or None if the hitter is mostly missing."""
    start, end = window_bounds(hit_frames, index, fps)
    frames = _resample_indices(start, end, seq_len)

    pose_out = np.zeros((seq_len, 17, 3), dtype=np.float32)
    ball_out = np.zeros((seq_len, 3), dtype=np.float32)
    last_pose: FloatArray | None = None
    centres: list[FloatArray] = []
    scales: list[float] = []
    present = 0

    for i, frame in enumerate(frames):
        poses = keypoints_by_frame.get(int(frame), [])
        ids = player_ids_by_frame.get(int(frame), [])
        hitter = _hitter_at(poses, ids, hitter_id) if poses else None
        if hitter is not None:
            last_pose = hitter
            present += 1
        # Hold the last known skeleton through gaps: the tracker drops a frame
        # here and there, and a zeroed frame would read as a real pose at origin.
        current = hitter if hitter is not None else last_pose
        if current is None:
            continue
        normalised, centre, scale = normalise_pose(current)
        pose_out[i] = normalised
        centres.append(centre)
        scales.append(scale)

    if present < seq_len // 4 or not centres:
        return None  # the hitter is missing for most of the window

    # Ball in the hitter's frame of reference: the model needs "where the ball is
    # relative to this player", not absolute pixels.
    mean_centre = np.mean(np.stack(centres), axis=0)
    mean_scale = float(np.mean(scales))
    for i, frame in enumerate(frames):
        position = ball_by_frame.get(int(frame))
        if position is None:
            continue
        ball_out[i, 0] = (position[0] - mean_centre[0]) / mean_scale
        ball_out[i, 1] = (position[1] - mean_centre[1]) / mean_scale
        ball_out[i, 2] = 1.0  # present flag: zeros elsewhere are "unknown"

    return ShotWindow(
        pose=pose_out,
        ball=ball_out,
        label=label,
        rally=rally,
        tournament=tournament,
        hit_frame=hit_frames[index],
    )


def _frame_states(
    keypoints_by_frame: dict[int, list[FloatArray]],
    player_ids_by_frame: dict[int, list[int]],
    ball_by_frame: dict[int, tuple[float, float]],
    n_frames: int,
) -> dict[int, FrameState]:
    """Adapt the cached arrays to what `assign_hit` expects."""
    from padel_cv.pipeline import BallDetection, PoseDetection
    from padel_ml.hit_assignment import FrameState

    states: dict[int, FrameState] = {}
    for frame in range(n_frames):
        poses = []
        for keypoints, player_id in zip(
            keypoints_by_frame.get(frame, []), player_ids_by_frame.get(frame, []), strict=True
        ):
            if player_id < 0:
                continue
            xs, ys = keypoints[:, 0], keypoints[:, 1]
            poses.append(
                PoseDetection(
                    bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    confidence=1.0,
                    keypoints=keypoints,
                    player_id=int(player_id),
                )
            )
        position = ball_by_frame.get(frame)
        states[frame] = FrameState(
            poses=poses,
            ball=BallDetection(position, 1.0) if position is not None else None,
        )
    return states


def build_rally_windows(
    features_npz: Path,
    labels_csv: Path,
    seq_len: int = SEQ_LEN,
) -> tuple[list[ShotWindow], int]:
    """Windows for one rally; also returns how many hits had no hitter assigned."""
    from padel_ml.hit_assignment import assign_hit

    data = np.load(features_npz, allow_pickle=True)
    keypoints_by_frame = data["keypoints"].item()
    player_ids_by_frame = data["player_ids"].item()
    ball_by_frame = data["ball"].item()
    fps = float(data["fps"])
    n_frames = int(data["n_frames"])

    labels = load_labels(labels_csv)
    hit_frames = [frame for frame, _ in labels]
    states = _frame_states(keypoints_by_frame, player_ids_by_frame, ball_by_frame, n_frames)

    rally = features_npz.stem
    tournament = "_".join(rally.split("_")[:2])
    windows: list[ShotWindow] = []
    unassigned = 0
    for index, (frame, shot_type) in enumerate(labels):
        if shot_type == DISCARD:
            continue  # kept in the CSV, never trained on (ADR-0016 D)
        hitter = assign_hit(frame, states)
        if hitter is None or hitter < 0:
            unassigned += 1
            continue
        window = build_window(
            keypoints_by_frame,
            player_ids_by_frame,
            ball_by_frame,
            hit_frames,
            index,
            hitter,
            fps,
            CLASS_TO_INDEX[shot_type],
            rally,
            tournament,
            seq_len,
        )
        if window is None:
            unassigned += 1
            continue
        windows.append(window)
    return windows, unassigned


def build_dataset(
    features_dir: Path,
    labels_dir: Path,
    seq_len: int = SEQ_LEN,
) -> tuple[list[ShotWindow], dict[str, int]]:
    """Every window across every rally that has both features and labels."""
    windows: list[ShotWindow] = []
    stats = {"rallies": 0, "unassigned": 0, "missing_features": 0}
    for labels_csv in sorted(labels_dir.glob("*.csv")):
        features = features_dir / f"{labels_csv.stem}.npz"
        if not features.exists():
            stats["missing_features"] += 1
            continue
        rally_windows, unassigned = build_rally_windows(features, labels_csv, seq_len)
        windows.extend(rally_windows)
        stats["rallies"] += 1
        stats["unassigned"] += unassigned
    return windows, stats

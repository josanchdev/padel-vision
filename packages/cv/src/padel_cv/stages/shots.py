"""Dummy shot detection stage (Level 1): wrist-speed peaks.

A stroke shows up in the skeleton as a sharp spike in wrist speed. This stage
tracks, per player, the fastest wrist in each frame; speed is normalized by
the player's torso length so near and far players are comparable (raw pixel
speed would make far players look slow). A shot is emitted when the smoothed
speed peaks above a threshold, with a refractory period so one stroke does not
fire multiple events.

This is deliberately simple: it exists to define and exercise the ShotEvent
contract end-to-end. The Level-2 skeleton classifier (ST-GCN or similar)
replaces the heuristic, not the interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from padel_cv.pipeline import Frame, PoseDetection, ShotEvent

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
MIN_KEYPOINT_CONFIDENCE = 0.3


def torso_length_px(pose: PoseDetection) -> float | None:
    """Shoulder-center to hip-center distance; the player's natural scale."""
    parts = pose.keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
    if (parts[:, 2] < MIN_KEYPOINT_CONFIDENCE).any():
        return None
    shoulders = parts[:2, :2].mean(axis=0)
    hips = parts[2:, :2].mean(axis=0)
    length = float(np.linalg.norm(shoulders - hips))
    return length if length > 1.0 else None


def wrist_positions(pose: PoseDetection) -> list[tuple[float, float]]:
    wrists = pose.keypoints[[LEFT_WRIST, RIGHT_WRIST]]
    return [(float(x), float(y)) for x, y, conf in wrists if conf >= MIN_KEYPOINT_CONFIDENCE]


@dataclass
class _PlayerState:
    last_wrists: list[tuple[float, float]]
    last_frame: int
    speeds: deque[tuple[int, float]]  # (frame_index, normalized speed)
    last_shot_frame: int = -(10**9)


class DummyShotStage:
    """Emits ShotEvents from normalized wrist-speed peaks, per player."""

    def __init__(
        self,
        speed_threshold: float = 0.6,
        smooth_window: int = 3,
        refractory_frames: int = 20,
        max_frame_gap: int = 5,
    ) -> None:
        # speed_threshold is in torso-lengths per frame: 0.6 means the wrist
        # moved more than half a torso length between consecutive frames.
        self._threshold = speed_threshold
        self._smooth = smooth_window
        self._refractory = refractory_frames
        self._max_gap = max_frame_gap
        self._players: dict[int, _PlayerState] = {}

    def process(self, frame: Frame) -> Frame:
        for pose in frame.poses:
            if pose.player_id is None:
                continue
            wrists = wrist_positions(pose)
            torso = torso_length_px(pose)
            if not wrists or torso is None:
                continue
            state = self._players.get(pose.player_id)
            if state is None or frame.index - state.last_frame > self._max_gap:
                self._players[pose.player_id] = _PlayerState(
                    last_wrists=wrists, last_frame=frame.index, speeds=deque(maxlen=16)
                )
                continue
            gap = frame.index - state.last_frame
            if gap <= 0:
                # Duplicate player_id in the same frame: ignore the extra pose.
                continue
            # Fastest wrist, matched to its nearest previous position.
            speed = max(
                min(float(np.hypot(x - px, y - py)) for px, py in state.last_wrists)
                for x, y in wrists
            ) / (torso * gap)
            state.speeds.append((frame.index, speed))
            state.last_wrists = wrists
            state.last_frame = frame.index
            self._maybe_emit(frame, pose.player_id, state)
        return frame

    def _maybe_emit(self, frame: Frame, player_id: int, state: _PlayerState) -> None:
        if len(state.speeds) < 2 * self._smooth + 1:
            return
        recent = list(state.speeds)[-(2 * self._smooth + 1) :]
        center_frame, _ = recent[self._smooth]
        if center_frame - state.last_shot_frame < self._refractory:
            return
        values = [s for _, s in recent]
        center = values[self._smooth]
        # Local maximum above threshold, evaluated with a small delay so we
        # can see both sides of the peak.
        if center >= self._threshold and center == max(values):
            state.last_shot_frame = center_frame
            frame.shot_events.append(
                ShotEvent(
                    player_id=player_id,
                    frame_index=center_frame,
                    label="shot",
                    confidence=min(center / (2 * self._threshold), 1.0),
                )
            )

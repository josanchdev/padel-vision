"""Real shot recognition stage: wrist-speed proposal + PoseConv3D classifier.

Two-stage design. The wrist-speed peak (our own signal, no external annotation)
proposes *when* a shot may happen — high recall, low precision. For each
proposal we crop the hitter's centered skeleton window and let PoseConv3D say
*what* it is, or reject it as NoShot. This replaces DummyShotStage: same
ShotEvent contract, real labels, and the classifier filters the false peaks.

Streaming detail: a peak at frame F needs frames after F to form a centered
window, so classification is deferred until the buffer reaches F + WINDOW/2.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from padel_cv.clip_builder import WINDOW
from padel_cv.pipeline import Frame, ShotEvent
from padel_cv.stages.shots import torso_length_px, wrist_positions
from padel_ml.classifier import ShotClassifier

_HALF = WINDOW // 2


@dataclass
class _PlayerBuffer:
    skeletons: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (17,3)
    speeds: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=16))
    last_wrists: list[tuple[float, float]] | None = None
    last_frame: int = -(10**9)
    last_shot_frame: int = -(10**9)
    pending: list[int] = field(default_factory=list)  # impact frames awaiting a window


class ClassifiedShotStage:
    """Wrist-speed proposals classified by PoseConv3D into real shot types."""

    def __init__(
        self,
        checkpoint: Path,
        speed_threshold: float = 0.5,
        smooth_window: int = 3,
        refractory_frames: int = 20,
        min_confidence: float = 0.4,
    ) -> None:
        self._classifier = ShotClassifier(checkpoint)
        self._threshold = speed_threshold
        self._smooth = smooth_window
        self._refractory = refractory_frames
        self._min_conf = min_confidence
        self._buffers: dict[int, _PlayerBuffer] = defaultdict(_PlayerBuffer)

    def process(self, frame: Frame) -> Frame:
        for pose in frame.poses:
            if pose.player_id is None:
                continue
            buf = self._buffers[pose.player_id]
            buf.skeletons[frame.index] = pose.keypoints
            self._update_speed(frame.index, pose, buf)
        self._classify_ready(frame)
        self._evict_old(frame.index)
        return frame

    def _update_speed(self, index: int, pose, buf: _PlayerBuffer) -> None:  # type: ignore[no-untyped-def]
        wrists = wrist_positions(pose)
        torso = torso_length_px(pose)
        if not wrists or torso is None:
            return
        gap = index - buf.last_frame
        if buf.last_wrists is not None and 0 < gap <= 5:
            speed = max(
                min(float(np.hypot(x - px, y - py)) for px, py in buf.last_wrists)
                for x, y in wrists
            ) / (torso * gap)
            buf.speeds.append((index, speed))
            self._detect_peak(buf)
        buf.last_wrists = wrists
        buf.last_frame = index

    def _detect_peak(self, buf: _PlayerBuffer) -> None:
        if len(buf.speeds) < 2 * self._smooth + 1:
            return
        recent = list(buf.speeds)[-(2 * self._smooth + 1) :]
        center_frame, _ = recent[self._smooth]
        if center_frame - buf.last_shot_frame < self._refractory:
            return
        values = [s for _, s in recent]
        if values[self._smooth] >= self._threshold and values[self._smooth] == max(values):
            buf.last_shot_frame = center_frame
            buf.pending.append(center_frame)

    def _classify_ready(self, frame: Frame) -> None:
        for player_id, buf in self._buffers.items():
            still_pending = []
            for impact in buf.pending:
                if frame.index < impact + _HALF:
                    still_pending.append(impact)
                    continue
                window = self._window(buf, impact)
                if window is None:
                    continue
                label, prob = self._classifier.classify_window(window)
                if self._classifier.is_shot(label) and prob >= self._min_conf:
                    frame.shot_events.append(
                        ShotEvent(
                            player_id=player_id, frame_index=impact, label=label, confidence=prob
                        )
                    )
            buf.pending = still_pending

    def _window(self, buf: _PlayerBuffer, impact: int) -> np.ndarray | None:
        """Centered (WINDOW, 17, 3) window with hold-fill for missing frames."""
        frames = range(impact - _HALF, impact - _HALF + WINDOW)
        present = [f for f in frames if f in buf.skeletons]
        if len(present) < WINDOW // 2:
            return None
        out, last = [], None
        for f in frames:
            if f in buf.skeletons:
                last = buf.skeletons[f]
            out.append(last if last is not None else buf.skeletons[present[0]])
        return np.stack(out).astype(np.float32)

    def _evict_old(self, index: int, keep: int = WINDOW * 2) -> None:
        for buf in self._buffers.values():
            for f in [f for f in buf.skeletons if f < index - keep]:
                del buf.skeletons[f]

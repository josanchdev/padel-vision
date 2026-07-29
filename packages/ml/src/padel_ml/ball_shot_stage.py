"""Shot recognition from the BALL, not wrist speed (ADR-0012).

Replaces the wrist-speed proposal (which fired on any brisk arm gesture) with the
physical cause of a shot: the ball sharply changes direction AND a player's wrist
is near it. Each ball direction-change is a shot (wrist near) or a bounce (no
player). The shot's type still comes from the PoseConv3D classifier (stage 2).

Streaming: ball events need a few frames of lookahead (to measure the outgoing
velocity), and the classifier needs a centered skeleton window, so both are
deferred until the buffer has caught up — same pattern as the old stage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from padel_cv.ball_events import BallSample, classify_events, detect_direction_changes
from padel_cv.clip_builder import WINDOW
from padel_cv.pipeline import Frame, PoseDetection, ShotEvent
from padel_cv.stages.shots import torso_length_px, wrist_positions
from padel_ml.classifier import ShotClassifier

_HALF = WINDOW // 2
_LOOKAHEAD = 3  # frames of trajectory needed after an impact to confirm the turn


@dataclass
class _PlayerBuffer:
    skeletons: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (17,3)


class BallShotStage:
    """Ball direction-change → shot (wrist near) → PoseConv3D type."""

    def __init__(
        self,
        checkpoint: Path,
        min_turn_deg: float = 45.0,
        wrist_dist_frac: float = 1.2,
        min_confidence: float = 0.4,
    ) -> None:
        self._classifier = ShotClassifier(checkpoint)
        self._min_turn = min_turn_deg
        # A wrist counts as "near the ball" within this many torso-lengths.
        self._wrist_dist_frac = wrist_dist_frac
        self._min_conf = min_confidence
        self._buffers: dict[int, _PlayerBuffer] = defaultdict(_PlayerBuffer)
        self._ball: deque[BallSample] = deque(maxlen=64)
        self._poses_at: dict[int, list[PoseDetection]] = {}  # frame -> on-court poses
        self._done: set[int] = set()  # impact frames already emitted

    def process(self, frame: Frame) -> Frame:
        # Record skeletons (for the classifier window) and ball trajectory.
        poses_here: list[PoseDetection] = []
        for pose in frame.poses:
            if pose.player_id is None:
                continue
            self._buffers[pose.player_id].skeletons[frame.index] = pose.keypoints
            poses_here.append(pose)
        self._poses_at[frame.index] = poses_here
        if frame.ball is not None:
            self._ball.append(BallSample(frame.index, *frame.ball.image_xy))

        self._detect_and_classify(frame)
        self._evict_old(frame.index)
        return frame

    def _detect_and_classify(self, frame: Frame) -> None:
        samples = list(self._ball)
        if len(samples) < 2 * _LOOKAHEAD + 1:
            return
        raw = detect_direction_changes(samples, window=_LOOKAHEAD, min_turn_deg=self._min_turn)
        events = classify_events(samples, raw, self._wrist_near)
        for ev in events:
            impact = ev.frame_index
            # Only shots become ShotEvents; only once the centered window exists.
            if ev.kind != "shot" or ev.player_id is None or impact in self._done:
                continue
            if frame.index < impact + _HALF:
                continue
            window = self._window(self._buffers[ev.player_id], impact)
            if window is None:
                self._done.add(impact)
                continue
            label, prob = self._classifier.classify_window(window)
            self._done.add(impact)
            if self._classifier.is_shot(label) and prob >= self._min_conf:
                frame.shot_events.append(
                    ShotEvent(
                        player_id=ev.player_id, frame_index=impact, label=label, confidence=prob
                    )
                )

    def _wrist_near(self, frame_index: int, ball_xy: tuple[float, float]) -> int | None:
        """The player whose wrist is closest to the ball at this frame, if close
        enough (within wrist_dist_frac torso-lengths); else None."""
        best_pid: int | None = None
        best_dist = float("inf")
        for pose in self._poses_at.get(frame_index, []):
            wrists = wrist_positions(pose)
            torso = torso_length_px(pose)
            if not wrists or torso is None or pose.player_id is None:
                continue
            for wx, wy in wrists:
                d = float(np.hypot(wx - ball_xy[0], wy - ball_xy[1])) / torso
                if d < best_dist:
                    best_dist, best_pid = d, pose.player_id
        return best_pid if best_dist <= self._wrist_dist_frac else None

    def _window(self, buf: _PlayerBuffer, impact: int) -> np.ndarray | None:
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

    def _evict_old(self, index: int, keep: int = WINDOW * 3) -> None:
        for buf in self._buffers.values():
            for f in [f for f in buf.skeletons if f < index - keep]:
                del buf.skeletons[f]
        for f in [f for f in self._poses_at if f < index - keep]:
            del self._poses_at[f]

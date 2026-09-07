"""Stable player identity 1-4, replicating Decorte 2024 (§5.2) — see ADR-0015.

Three pieces, in the order the paper applies them:

1. **Court mask.** Trackers happily follow spectators. Poses are kept only if
   their bbox overlaps a padded trapezoid built from the court corners.
2. **Initial assignment.** Average each track's bbox centre over the first 3
   seconds of the rally, then sort: leftmost pair vs rightmost pair by x, and
   within each pair by y. Top team gets 1-2 (left to right), bottom team 3-4.
   This is the numbering the dataset's hit annotations use.
3. **Re-identification.** Padel players wear team kit, so visual re-id (DeepSORT
   and friends) has little to work with. The paper instead exploits that the
   player count is known (4) and that both players of a side are rarely lost at
   once: when a track dies and a new one appears, the new track inherits the
   slot that just went missing. We disambiguate by nearest last-known position.

This replaces the court-half anchoring of `stages/identity.py`, which needed a
homography and could not recover a slot after both teammates were lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

from padel_cv.pipeline import PoseDetection

INITIAL_WINDOW_S = 3.0
"""Paper: players are numbered from their average position over the rally's
first three seconds."""

MASK_PADDING_PX = 60.0
"""Trapezoid padding so players stepping just outside the lines still count."""

MAX_MISSING_FRAMES = 75
"""How long a slot stays claimable by a new track after its own track died."""

MIN_MISSING_FRAMES = 2
"""A slot must be absent this long before a new track may claim it: one dropped
detection is a flicker, not a lost player."""


def _bbox_centre(pose: PoseDetection) -> tuple[float, float]:
    x1, y1, x2, y2 = pose.bbox_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def court_mask_polygon(
    corners_px: npt.NDArray[np.float64], padding_px: float = MASK_PADDING_PX
) -> npt.NDArray[np.float64]:
    """Expand the four court corners outwards from their centroid by `padding_px`.

    Players routinely stand behind the baseline or wide of the sidelines (padel
    is played off the walls), so the raw court quad is too tight a filter.
    """
    centroid = corners_px.mean(axis=0)
    out = []
    for corner in corners_px:
        direction = corner - centroid
        norm = float(np.hypot(*direction))
        out.append(corner + direction / norm * padding_px if norm > 1e-6 else corner)
    return np.array(out, dtype=np.float64)


def is_on_court_region(pose: PoseDetection, polygon: npt.NDArray[np.float64]) -> bool:
    """Whether a pose belongs to a player rather than a spectator.

    Tests the bbox's bottom-centre (where the player stands) against the padded
    court polygon — the feet are what tell a player from someone in the stands.
    """
    x1, _, x2, y2 = pose.bbox_xyxy
    point = ((x1 + x2) / 2, y2)
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def filter_players(
    poses: list[PoseDetection], polygon: npt.NDArray[np.float64] | None
) -> list[PoseDetection]:
    """Drop detections outside the court region (spectators, ball kids, umpire)."""
    if polygon is None:
        return poses
    return [p for p in poses if is_on_court_region(p, polygon)]


def assign_initial_slots(
    positions_by_track: dict[int, tuple[float, float]],
) -> dict[int, int]:
    """Number four tracks 1-4 by starting position (paper §5.2).

    Sort by x to split left from right, then by y within each side: the top-half
    (further from camera) player of the left pair is 1, of the right pair 2; the
    bottom-half ones are 3 and 4. Matches the dataset's annotation scheme.
    """
    if len(positions_by_track) < 4:
        return {}
    by_x = sorted(positions_by_track.items(), key=lambda kv: kv[1][0])
    left, right = by_x[:2], by_x[2:4]
    # within each side, smaller y = further from the camera = top team
    left_top, left_bottom = sorted(left, key=lambda kv: kv[1][1])
    right_top, right_bottom = sorted(right, key=lambda kv: kv[1][1])
    return {
        left_top[0]: 1,
        right_top[0]: 2,
        left_bottom[0]: 3,
        right_bottom[0]: 4,
    }


@dataclass
class PlayerIdentityTracker:
    """Maps volatile tracker IDs onto the four stable player slots.

    Feed it one frame at a time via `update`; it collects the initial window,
    fixes the 1-4 numbering, and thereafter re-identifies new tracks by which
    slot recently went missing and where it was last seen.
    """

    fps: float = 25.0
    initial_window_s: float = INITIAL_WINDOW_S
    max_missing_frames: int = MAX_MISSING_FRAMES
    min_missing_frames: int = MIN_MISSING_FRAMES

    slot_by_track: dict[int, int] = field(default_factory=dict)
    last_position: dict[int, tuple[float, float]] = field(default_factory=dict)
    last_seen_frame: dict[int, int] = field(default_factory=dict)
    _initial_samples: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    _initial_poses: list[PoseDetection] = field(default_factory=list)
    _initialized: bool = False

    def update(self, frame_index: int, poses: list[PoseDetection]) -> None:
        """Assign `player_id` on every pose of this frame (in place)."""
        if not self._initialized:
            self._collect_initial(frame_index, poses)
            if frame_index < int(self.initial_window_s * self.fps):
                return
            self._finish_initial()
            self._label_initial_window()

        seen_now: list[PoseDetection] = []
        unknown: list[PoseDetection] = []
        for pose in poses:
            if pose.track_id is None:
                continue
            slot = self.slot_by_track.get(pose.track_id)
            if slot is None:
                unknown.append(pose)
            else:
                pose.player_id = slot
                self.last_position[slot] = _bbox_centre(pose)
                self.last_seen_frame[slot] = frame_index
                seen_now.append(pose)

        if unknown:
            self._reidentify(frame_index, unknown, {p.player_id for p in seen_now})

    def _collect_initial(self, frame_index: int, poses: list[PoseDetection]) -> None:
        for pose in poses:
            if pose.track_id is not None:
                self._initial_samples.setdefault(pose.track_id, []).append(_bbox_centre(pose))
                self._initial_poses.append(pose)

    def _label_initial_window(self) -> None:
        """Back-fill IDs on the poses of the startup window.

        The first three seconds are what *defines* the numbering, but they are
        still part of the rally — short rallies have hits in them — so once the
        slots are known they are applied to those poses too.
        """
        for pose in self._initial_poses:
            if pose.track_id is not None:
                pose.player_id = self.slot_by_track.get(pose.track_id)
        self._initial_poses.clear()

    def _finish_initial(self) -> None:
        """Pick the four longest-lived tracks of the window and number them."""
        best = sorted(self._initial_samples.items(), key=lambda kv: len(kv[1]), reverse=True)[:4]
        averages = {
            track: (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            )
            for track, pts in best
        }
        self.slot_by_track = assign_initial_slots(averages)
        for track, slot in self.slot_by_track.items():
            self.last_position[slot] = averages[track]
        self._initialized = True

    def _reidentify(
        self, frame_index: int, unknown: list[PoseDetection], slots_seen: set[int | None]
    ) -> None:
        """Give each new track the missing slot whose last position is nearest.

        The paper's reasoning: with four known players, a track that appears
        while exactly one slot is unaccounted for must be that player.
        """
        # A slot is claimable only if its own track has really been gone for a
        # few frames (a single dropped detection is not a lost player) and not
        # so long that the player has surely left. Slots held by a live track
        # are never reassigned, or two tracks would share one player.
        live_slots = set(self.slot_by_track.values()) & {
            s for s in (1, 2, 3, 4) if frame_index - self.last_seen_frame.get(s, -(10**9)) <= 1
        }
        missing = [
            slot
            for slot in (1, 2, 3, 4)
            if slot not in slots_seen
            and slot not in live_slots
            and self.min_missing_frames
            <= frame_index - self.last_seen_frame.get(slot, -(10**9))
            <= self.max_missing_frames
        ]
        for pose in unknown:
            if not missing:
                return
            centre = _bbox_centre(pose)
            slot = min(
                missing,
                key=lambda s: float(np.hypot(*(np.array(centre) - self.last_position[s])))
                if s in self.last_position
                else float("inf"),
            )
            missing.remove(slot)
            assert pose.track_id is not None
            # a slot belongs to exactly one track: drop any stale claim on it
            for track, held in list(self.slot_by_track.items()):
                if held == slot:
                    del self.slot_by_track[track]
            self.slot_by_track[pose.track_id] = slot
            pose.player_id = slot
            self.last_position[slot] = centre
            self.last_seen_frame[slot] = frame_index

"""Stable player identity (J1-J4) from court position.

Trackers give volatile IDs: whenever a player is lost (occlusion, leaving the
frame between points) their track dies and a fresh ID is born. But padel has
exactly four players and each team stays on its own court half, so identity
can be anchored to geometry: slots 1-2 belong to the near half (y < 10 m),
slots 3-4 to the far half. When a track dies, its slot becomes stale and is
inherited by the next new track appearing on that half.

Known limitation (future work): teams swap court sides between games; slot
identity is stable within a side period, not across swaps.
"""

from __future__ import annotations

from padel_cv.court import NET_Y_M
from padel_cv.pipeline import Frame, PoseDetection

NEAR_SLOTS = (1, 2)
FAR_SLOTS = (3, 4)


class PlayerIdentityStage:
    """Maps tracker IDs to stable player slots using court-half geometry."""

    def __init__(self, stale_after_frames: int = 60) -> None:
        self._slot_by_track: dict[int, int] = {}
        self._slot_last_seen: dict[int, int] = {}
        self._stale_after = stale_after_frames

    def process(self, frame: Frame) -> Frame:
        candidates = [
            pose
            for pose in frame.poses
            if pose.on_court and pose.court_position_m is not None and pose.track_id is not None
        ]
        unassigned = []
        for pose in candidates:
            slot = self._slot_by_track.get(pose.track_id)  # type: ignore[arg-type]
            if slot is not None:
                pose.player_id = slot
                self._slot_last_seen[slot] = frame.index
            else:
                unassigned.append(pose)

        # New tracks claim stale slots on their court half. Left-to-right order
        # keeps assignment deterministic when several appear at once.
        for pose in sorted(unassigned, key=lambda p: p.court_position_m[0]):  # type: ignore[index]
            slots = self._half_slots(pose)
            free = [s for s in slots if self._is_stale(s, frame.index)]
            if not free:
                # Both teammates on this half are alive: extra detection
                # (mis-filtered spectator, transient duplicate) gets no slot.
                continue
            slot = free[0]
            self._slot_by_track[pose.track_id] = slot  # type: ignore[index]
            pose.player_id = slot
            self._slot_last_seen[slot] = frame.index
        return frame

    @staticmethod
    def _half_slots(pose: PoseDetection) -> tuple[int, int]:
        assert pose.court_position_m is not None
        return NEAR_SLOTS if pose.court_position_m[1] < NET_Y_M else FAR_SLOTS

    def _is_stale(self, slot: int, frame_index: int) -> bool:
        last_seen = self._slot_last_seen.get(slot)
        return last_seen is None or frame_index - last_seen > self._stale_after

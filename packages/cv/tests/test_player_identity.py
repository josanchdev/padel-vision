import numpy as np

from padel_cv.pipeline import PoseDetection
from padel_cv.player_identity import (
    PlayerIdentityTracker,
    assign_initial_slots,
    court_mask_polygon,
    filter_players,
    is_on_court_region,
)


def _pose(track_id, cx, cy, w=60, h=160):
    kp = np.zeros((17, 3), dtype=np.float32)
    return PoseDetection(
        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.9, kp, track_id=track_id
    )


COURT = np.array([[400.0, 300.0], [1500.0, 300.0], [1750.0, 950.0], [150.0, 950.0]])


def test_mask_polygon_expands_outwards() -> None:
    poly = court_mask_polygon(COURT, padding_px=50)
    # every corner sits 50 px further from the centroid than it did
    centroid = COURT.mean(axis=0)
    before = np.linalg.norm(COURT - centroid, axis=1)
    after = np.linalg.norm(poly - centroid, axis=1)
    assert np.allclose(after - before, 50.0)


def test_spectator_outside_court_is_filtered() -> None:
    poly = court_mask_polygon(COURT)
    player = _pose(1, 900, 800)  # feet inside the court
    spectator = _pose(2, 900, 120)  # up in the stands
    assert is_on_court_region(player, poly)
    assert not is_on_court_region(spectator, poly)
    assert filter_players([player, spectator], poly) == [player]


def test_initial_slots_follow_paper_numbering() -> None:
    # left-top, right-top, left-bottom, right-bottom
    slots = assign_initial_slots(
        {
            10: (500.0, 400.0),  # left, far  -> 1
            11: (1400.0, 420.0),  # right, far -> 2
            12: (450.0, 800.0),  # left, near -> 3
            13: (1450.0, 820.0),  # right,near -> 4
        }
    )
    assert slots == {10: 1, 11: 2, 12: 3, 13: 4}


def _run(tracker, frames):
    for i, poses in enumerate(frames):
        tracker.update(i, poses)


def test_tracker_assigns_and_keeps_slots() -> None:
    tracker = PlayerIdentityTracker(fps=25.0)
    layout = [(10, 500, 400), (11, 1400, 420), (12, 450, 800), (13, 1450, 820)]
    frames = [[_pose(t, x, y) for t, x, y in layout] for _ in range(80)]
    _run(tracker, frames)
    last = {p.track_id: p.player_id for p in frames[-1]}
    assert last == {10: 1, 11: 2, 12: 3, 13: 4}


def test_new_track_inherits_the_missing_slot() -> None:
    """Player 3's track dies and reappears under a new id -> still player 3."""
    tracker = PlayerIdentityTracker(fps=25.0)
    layout = [(10, 500, 400), (11, 1400, 420), (12, 450, 800), (13, 1450, 820)]
    frames = [[_pose(t, x, y) for t, x, y in layout] for _ in range(80)]
    # frames 80-84: track 12 (player 3) is lost
    for _ in range(5):
        frames.append([_pose(t, x, y) for t, x, y in layout if t != 12])
    # frame 85: a new track 99 appears where player 3 was
    frames.append([_pose(t, x, y) for t, x, y in layout if t != 12] + [_pose(99, 460, 805)])
    _run(tracker, frames)
    reborn = next(p for p in frames[-1] if p.track_id == 99)
    assert reborn.player_id == 3

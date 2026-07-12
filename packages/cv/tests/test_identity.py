import numpy as np

from padel_cv.pipeline import Frame, PoseDetection
from padel_cv.stages import PlayerIdentityStage


def make_frame(index: int, players: list[tuple[int, float, float]]) -> Frame:
    """players: (track_id, x_m, y_m) already on court."""
    frame = Frame(index=index, timestamp_s=index / 30.0, image=np.zeros((4, 4, 3), np.uint8))
    for track_id, x_m, y_m in players:
        frame.poses.append(
            PoseDetection(
                bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                confidence=0.9,
                keypoints=np.zeros((17, 3), np.float32),
                track_id=track_id,
                court_position_m=(x_m, y_m),
                on_court=True,
            )
        )
    return frame


def slots(frame: Frame) -> dict[int, int | None]:
    return {p.track_id: p.player_id for p in frame.poses if p.track_id is not None}


def test_four_players_get_four_slots_by_half_and_left_right_order() -> None:
    stage = PlayerIdentityStage()
    frame = stage.process(
        make_frame(0, [(10, 7.0, 5.0), (11, 3.0, 4.0), (12, 2.0, 15.0), (13, 8.0, 16.0)])
    )
    assert slots(frame) == {11: 1, 10: 2, 12: 3, 13: 4}  # near: left J1, right J2


def test_slot_survives_track_id_switch_after_staleness() -> None:
    stage = PlayerIdentityStage(stale_after_frames=5)
    stage.process(make_frame(0, [(10, 3.0, 5.0), (11, 7.0, 5.0)]))
    # Track 10 (slot J1) dies; only track 11 remains for a while.
    for i in range(1, 10):
        stage.process(make_frame(i, [(11, 7.0, 5.0)]))
    # A new track appears on the near half: it inherits the stale slot J1.
    frame = stage.process(make_frame(10, [(11, 7.0, 5.0), (99, 3.2, 5.5)]))
    assert slots(frame) == {11: 2, 99: 1}


def test_extra_detection_gets_no_slot_while_both_teammates_alive() -> None:
    stage = PlayerIdentityStage()
    stage.process(make_frame(0, [(10, 3.0, 5.0), (11, 7.0, 5.0)]))
    frame = stage.process(make_frame(1, [(10, 3.0, 5.0), (11, 7.0, 5.0), (50, 5.0, 8.0)]))
    assert slots(frame)[50] is None


def test_off_court_and_positionless_poses_are_ignored() -> None:
    stage = PlayerIdentityStage()
    frame = make_frame(0, [(10, 3.0, 5.0)])
    frame.poses[0].on_court = False
    frame = stage.process(frame)
    assert frame.poses[0].player_id is None

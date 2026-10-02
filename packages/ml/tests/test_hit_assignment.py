import numpy as np
from padel_ml.hit_assignment import FrameState, assign_hit, frame_states

from padel_cv.detections import BallDetection, PoseDetection


def _pose(pid: int, wrist_xy: tuple[float, float]) -> PoseDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[9] = [wrist_xy[0], wrist_xy[1], 0.9]  # left wrist
    kp[10] = [wrist_xy[0] + 5, wrist_xy[1], 0.9]
    return PoseDetection((0, 0, 10, 10), 0.9, kp, player_id=pid)


def _states(hit_frame, ball_xy, p1_wrist, p2_wrist, span=(-6, 7)):
    out = {}
    for f in range(hit_frame + span[0], hit_frame + span[1]):
        out[f] = FrameState(
            poses=[_pose(1, p1_wrist), _pose(2, p2_wrist)],
            ball=BallDetection(image_xy=ball_xy, confidence=0.9),
        )
    return out


def test_assigns_closest_player() -> None:
    # ball next to player 2's wrist across the window
    st = _states(100, ball_xy=(500.0, 300.0), p1_wrist=(100.0, 100.0), p2_wrist=(505.0, 300.0))
    assert assign_hit(100, st) == 2


def test_robust_to_frames_without_ball() -> None:
    st = _states(100, (500.0, 300.0), (100.0, 100.0), (505.0, 300.0))
    # knock out the ball on several frames — vote should still land on player 2
    for f in (98, 99, 101, 102):
        st[f] = FrameState(poses=st[f].poses, ball=None)
    assert assign_hit(100, st) == 2


def test_none_when_no_data() -> None:
    assert assign_hit(100, {}) is None


def test_switches_with_the_ball() -> None:
    # now the ball is by player 1
    st = _states(100, ball_xy=(100.0, 100.0), p1_wrist=(102.0, 100.0), p2_wrist=(505.0, 300.0))
    assert assign_hit(100, st) == 1


def test_team_alternation_fills_a_gap() -> None:
    from padel_ml.hit_assignment import team_alternation_sweep

    # players 1,2 = team 1; players 3,4 = team 2. Gap sandwiched by team 1.
    out = team_alternation_sweep({10: 1, 20: None, 30: 2})
    assert out[20] == -2  # must have been team 2


def test_frame_states_carry_the_ball_only_where_it_was_seen() -> None:
    players = {0: [_pose(1, (10, 10))], 1: [_pose(2, (20, 20))]}
    states = frame_states(players, {1: (21.0, 20.0)})
    assert set(states) == {0, 1}
    assert states[0].ball is None
    assert states[1].ball is not None and states[1].ball.image_xy == (21.0, 20.0)
    assert [p.player_id for p in states[1].poses] == [2]

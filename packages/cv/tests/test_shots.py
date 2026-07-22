import numpy as np

from padel_cv.pipeline import Frame, PoseDetection
from padel_cv.stages import DummyShotStage
from padel_cv.stages.shots import torso_length_px, wrist_positions


def make_pose(wrist_x: float, player_id: int = 1) -> PoseDetection:
    """Upright skeleton with a 100 px torso; only the right wrist moves."""
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = [480.0, 300.0, 0.9]  # left shoulder
    keypoints[6] = [520.0, 300.0, 0.9]  # right shoulder
    keypoints[11] = [485.0, 400.0, 0.9]  # left hip
    keypoints[12] = [515.0, 400.0, 0.9]  # right hip
    keypoints[9] = [470.0, 350.0, 0.9]  # left wrist (static)
    keypoints[10] = [wrist_x, 350.0, 0.9]  # right wrist (moves)
    return PoseDetection(
        bbox_xyxy=(450.0, 250.0, 550.0, 500.0),
        confidence=0.9,
        keypoints=keypoints,
        player_id=player_id,
    )


def run_sequence(stage: DummyShotStage, wrist_xs: list[float]) -> list[int]:
    """Feed a wrist trajectory; return frame indices of emitted shots."""
    shots = []
    for i, wrist_x in enumerate(wrist_xs):
        frame = Frame(index=i, timestamp_s=i / 30.0, image=np.zeros((2, 2, 3), np.uint8))
        frame.poses = [make_pose(wrist_x)]
        frame = stage.process(frame)
        shots.extend(e.frame_index for e in frame.shot_events)
    return shots


def test_torso_and_wrists_helpers() -> None:
    pose = make_pose(530.0)
    assert torso_length_px(pose) == 100.0
    assert len(wrist_positions(pose)) == 2


def test_swing_triggers_one_shot() -> None:
    # Static wrist, then a violent 2-frame swing (150 px/frame = 1.5 torsos), then static.
    xs = [530.0] * 10 + [680.0, 830.0] + [830.0] * 10
    shots = run_sequence(DummyShotStage(), xs)
    assert len(shots) == 1
    assert 9 <= shots[0] <= 13  # at/near the swing


def test_slow_movement_never_triggers() -> None:
    # Drift of 10 px/frame = 0.1 torsos/frame, well under threshold.
    xs = [530.0 + 10.0 * i for i in range(30)]
    assert run_sequence(DummyShotStage(), xs) == []


def test_refractory_period_separates_shots() -> None:
    swing = [530.0] * 12 + [680.0, 830.0] + [830.0] * 12
    back = [830.0] * 12 + [680.0, 530.0] + [530.0] * 12
    shots = run_sequence(DummyShotStage(refractory_frames=20), swing + back)
    assert len(shots) == 2
    assert shots[1] - shots[0] >= 20


def test_second_swing_inside_refractory_is_suppressed() -> None:
    swing = [530.0] * 8 + [680.0, 830.0] + [830.0] * 8
    back = [830.0] * 8 + [680.0, 530.0] + [530.0] * 8
    assert len(run_sequence(DummyShotStage(refractory_frames=20), swing + back)) == 1


def test_players_without_id_are_ignored() -> None:
    stage = DummyShotStage()
    frame = Frame(index=0, timestamp_s=0.0, image=np.zeros((2, 2, 3), np.uint8))
    pose = make_pose(530.0)
    pose.player_id = None
    frame.poses = [pose]
    assert stage.process(frame).shot_events == []

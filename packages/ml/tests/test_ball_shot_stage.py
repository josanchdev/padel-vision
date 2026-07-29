import numpy as np
import torch
from padel_ml.poseconv3d import PoseConv3D

from padel_cv.pipeline import BallDetection, Frame, PoseDetection


def _save_ckpt(path, classes) -> None:
    model = PoseConv3D(num_classes=len(classes))
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)


def _pose(player_id, wrist_xy) -> PoseDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [480, 300, 0.9]  # shoulders / hips give a torso length
    kp[6] = [520, 300, 0.9]
    kp[11] = [485, 400, 0.9]
    kp[12] = [515, 400, 0.9]
    kp[9] = [wrist_xy[0], wrist_xy[1], 0.9]  # left wrist at the ball
    kp[10] = [wrist_xy[0] + 5, wrist_xy[1], 0.9]
    p = PoseDetection((450, 250, 550, 500), 0.9, kp, player_id=player_id)
    return p


def test_shot_emitted_when_ball_turns_near_a_wrist(tmp_path) -> None:
    from padel_ml.ball_shot_stage import BallShotStage

    ckpt = tmp_path / "m.pt"
    _save_ckpt(ckpt, ["Forehand", "Backhand", "Smash", "Serve", "Other", "NoShot"])
    stage = BallShotStage(ckpt, min_confidence=0.0)  # accept any class

    # Ball flies in, sharply reverses near frame 8 where J1's wrist sits, then
    # keeps going long enough (30 frames) for the centered classifier window.
    xs = [(300 + 20 * i, 300 - 20 * i) for i in range(9)]  # in: up-right to (460,140)
    xs += [(460 - 20 * i, 140 + 20 * i) for i in range(1, 30)]  # out: reversed, long tail
    events = []
    for i, (bx, by) in enumerate(xs):
        f = Frame(index=i, timestamp_s=i / 30, image=np.zeros((2, 2, 3), np.uint8))
        # J1's wrist is at the ball around the swing (frames 6-10); far otherwise.
        wrist = (bx, by) if 6 <= i <= 10 else (10_000, 10_000)
        f.poses = [_pose(1, wrist)]
        f.ball = BallDetection(image_xy=(bx, by), confidence=0.9)
        f = stage.process(f)
        events.extend(f.shot_events)

    assert events, "a ball reversal next to a wrist should yield a shot"
    assert all(e.player_id == 1 for e in events)
    assert all(e.label != "NoShot" for e in events)


def test_no_shot_when_ball_goes_straight(tmp_path) -> None:
    from padel_ml.ball_shot_stage import BallShotStage

    ckpt = tmp_path / "m.pt"
    _save_ckpt(ckpt, ["Forehand", "NoShot"])
    stage = BallShotStage(ckpt, min_confidence=0.0)

    events = []
    for i in range(20):
        f = Frame(index=i, timestamp_s=i / 30, image=np.zeros((2, 2, 3), np.uint8))
        f.poses = [_pose(1, (100 + 10 * i, 100))]  # wrist follows, but ball is straight
        f.ball = BallDetection(image_xy=(100 + 10 * i, 100), confidence=0.9)
        f = stage.process(f)
        events.extend(f.shot_events)
    assert events == [], "a straight ball has no direction change -> no shot"

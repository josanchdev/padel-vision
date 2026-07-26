import numpy as np
import torch
from padel_ml.poseconv3d import PoseConv3D

from padel_cv.pipeline import Frame, PoseDetection


def make_pose(player_id: int, wrist_x: float) -> PoseDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [480, 300, 0.9]
    kp[6] = [520, 300, 0.9]
    kp[11] = [485, 400, 0.9]
    kp[12] = [515, 400, 0.9]
    kp[9] = [470, 350, 0.9]
    kp[10] = [wrist_x, 350, 0.9]
    return PoseDetection((450, 250, 550, 500), 0.9, kp, player_id=player_id)


def save_fake_checkpoint(path, classes) -> None:
    model = PoseConv3D(num_classes=len(classes))
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)


def test_stage_emits_classified_event_on_swing(tmp_path) -> None:
    from padel_ml.shot_stage import ClassifiedShotStage

    ckpt = tmp_path / "model.pt"
    save_fake_checkpoint(ckpt, ["Forehand", "Backhand", "Smash", "Serve", "Other", "NoShot"])
    stage = ClassifiedShotStage(ckpt, min_confidence=0.0)  # accept any prediction

    # Static wrist, a sharp swing, then static — long enough to form a window.
    xs = [520.0] * 12 + [670.0, 820.0] + [820.0] * 30
    events = []
    for i, wx in enumerate(xs):
        frame = Frame(index=i, timestamp_s=i / 30, image=np.zeros((2, 2, 3), np.uint8))
        frame.poses = [make_pose(1, wx)]
        frame = stage.process(frame)
        events.extend(frame.shot_events)
    # A peak is proposed and classified into one of the shot classes (or NoShot,
    # which is filtered). With min_confidence=0 at least the proposal is reached.
    assert all(e.player_id == 1 for e in events)
    assert all(e.label != "NoShot" for e in events)


def test_stage_ignores_players_without_id(tmp_path) -> None:
    from padel_ml.shot_stage import ClassifiedShotStage

    ckpt = tmp_path / "model.pt"
    save_fake_checkpoint(ckpt, ["Forehand", "NoShot"])
    stage = ClassifiedShotStage(ckpt)
    frame = Frame(index=0, timestamp_s=0, image=np.zeros((2, 2, 3), np.uint8))
    pose = make_pose(1, 520.0)
    pose.player_id = None
    frame.poses = [pose]
    assert stage.process(frame).shot_events == []

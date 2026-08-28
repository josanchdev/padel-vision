import numpy as np
from padel_ml.shot_detect_dataset import (
    WINDOW,
    build_detect_windows,
    to_dataset,
)
from padel_ml.shot_eval import ShotBlock


def _person(wrist_xy: tuple[float, float]) -> np.ndarray:
    """A skeleton with a valid torso and a right wrist at wrist_xy."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [480, 300, 0.9]  # L shoulder
    kp[6] = [520, 300, 0.9]  # R shoulder
    kp[11] = [485, 400, 0.9]  # L hip
    kp[12] = [515, 400, 0.9]  # R hip
    kp[10] = [wrist_xy[0], wrist_xy[1], 0.9]  # R wrist
    return kp


def _scene(
    n_frames: int, ball_xy: tuple[float, float]
) -> tuple[dict[int, list[np.ndarray]], dict[int, tuple[float, float]]]:
    """One player whose wrist tracks the ball, ball present every frame."""
    persons = {f: [_person(ball_xy)] for f in range(n_frames)}
    ball = {f: ball_xy for f in range(n_frames)}
    return persons, ball


def test_positive_window_is_centred_and_shaped() -> None:
    persons, ball = _scene(60, (500.0, 350.0))
    blocks = [ShotBlock(start=24, end=32, category="Forehand")]  # centre 28
    wins = build_detect_windows(persons, ball, blocks, negatives_ratio=0.0)
    pos = [w for w in wins if w.label == 1]
    assert len(pos) == 1
    assert pos[0].source_frame == 28
    assert pos[0].pose.shape == (WINDOW, 17, 3)
    assert pos[0].ball.shape == (WINDOW, 3)


def test_ball_is_in_skeleton_normalized_frame() -> None:
    # _person has hips at y=400, x centred at 500 -> hip centre (500, 400).
    # A ball there normalizes to the origin, present=1.
    hip_center = (500.0, 400.0)
    persons = {f: [_person((500.0, 300.0))] for f in range(60)}
    ball = {f: hip_center for f in range(60)}
    blocks = [ShotBlock(24, 32, "Smash")]
    wins = build_detect_windows(persons, ball, blocks, negatives_ratio=0.0)
    ball_win = wins[0].ball
    assert np.allclose(ball_win[:, 2], 1.0)  # all present
    assert np.allclose(ball_win[:, :2], 0.0, atol=1e-4)  # ball at hip centre -> origin


def test_missing_ball_flagged_absent() -> None:
    persons = {f: [_person((500.0, 300.0))] for f in range(60)}
    ball = {f: (500.0, 350.0) for f in range(60) if f != 28}  # ball missing at centre
    blocks = [ShotBlock(24, 32, "Forehand")]
    wins = build_detect_windows(persons, ball, blocks, negatives_ratio=0.0)
    ball_win = wins[0].ball
    centre_idx = WINDOW // 2  # frame 28 is at the window centre
    assert ball_win[centre_idx, 2] == 0.0  # flagged absent
    assert ball_win[0, 2] == 1.0  # neighbours present


def test_negatives_are_far_from_shots() -> None:
    persons, ball = _scene(200, (500.0, 350.0))
    blocks = [ShotBlock(96, 104, "Forehand")]  # centre 100
    wins = build_detect_windows(persons, ball, blocks, negatives_ratio=3.0, min_gap=30, seed=1)
    negs = [w for w in wins if w.label == 0]
    assert negs, "should sample some negatives"
    assert all(abs(w.source_frame - 100) > 30 for w in negs)


def test_to_dataset_stacks_with_match_ids() -> None:
    persons, ball = _scene(80, (500.0, 350.0))
    b = [ShotBlock(36, 44, "Forehand")]
    m0 = build_detect_windows(persons, ball, b, negatives_ratio=1.0, seed=0)
    m1 = build_detect_windows(persons, ball, b, negatives_ratio=1.0, seed=2)
    ds = to_dataset([m0, m1])
    assert ds.pose.shape[1:] == (WINDOW, 17, 3)
    assert ds.ball.shape[1:] == (WINDOW, 3)
    assert set(ds.matches.tolist()) == {0, 1}
    assert len(ds.labels) == len(m0) + len(m1)


def test_subsample_renumbers_frames_and_blocks() -> None:
    from padel_ml.shot_detect_dataset import _subsample
    from padel_ml.shot_eval import ShotBlock

    persons = {f: [np.zeros((17, 3), np.float32)] for f in range(0, 10)}
    ball = {f: (float(f), 0.0) for f in range(0, 10)}
    blocks = [ShotBlock(4, 6, "Forehand")]  # centre 5 at 60fps -> centre 2 at 30fps
    p, b, bl = _subsample(persons, ball, blocks, step=2)
    # Every frame reindexed to f//2 (odd frames NOT dropped): 0..9 -> 0..4.
    assert set(p.keys()) == {0, 1, 2, 3, 4}
    assert bl[0].start == 2 and bl[0].end == 3
    # A ball marked on an ODD frame survives: frame 5 -> index 2, kept.
    assert b[2] == (4.0, 0.0)  # first frame mapping to index 2 (frame 4) wins


def test_subsample_step1_is_noop() -> None:
    from padel_ml.shot_detect_dataset import _subsample
    from padel_ml.shot_eval import ShotBlock

    persons = {0: [np.zeros((17, 3), np.float32)]}
    blocks = [ShotBlock(0, 2, "Smash")]
    p, _b, bl = _subsample(persons, {}, blocks, step=1)
    assert p == persons and bl == blocks


def test_dense_windows_label_per_frame() -> None:
    from padel_ml.shot_detect_dataset import WINDOW, build_dense_windows
    from padel_ml.shot_eval import ShotBlock

    # one player present across 200 frames, a shot at frame 100
    persons = {f: [np.zeros((17, 3), np.float32)] for f in range(200)}
    for kp in [persons[f][0] for f in range(200)]:
        kp[5] = [480, 300, 0.9]
        kp[6] = [520, 300, 0.9]
        kp[11] = [485, 400, 0.9]
        kp[12] = [515, 400, 0.9]
    ball = {f: (500.0, 350.0) for f in range(200)}
    blocks = [ShotBlock(100, 100, "Forehand")]
    wins = build_dense_windows(persons, ball, blocks, tol=1, stride=WINDOW)
    # the window centred on 100 must have 1s only around frame 100 (+-1)
    centred = next(w for w in wins if w.source_frame == 100)
    assert centred.labels.shape == (WINDOW,)
    assert centred.labels.sum() == 3  # frames 99,100,101
    # a window far from the shot is all zeros
    far = [w for w in wins if abs(w.source_frame - 100) > WINDOW]
    assert far and far[0].labels.sum() == 0


def test_dense_dataset_stacks_labels() -> None:
    from padel_ml.shot_detect_dataset import (
        WINDOW,
        build_dense_windows,
        to_dense_dataset,
    )
    from padel_ml.shot_eval import ShotBlock

    persons = {f: [_full_person()] for f in range(120)}
    ball = {f: (500.0, 400.0) for f in range(120)}
    b = [ShotBlock(60, 60, "Smash")]
    m0 = build_dense_windows(persons, ball, b, stride=WINDOW)
    ds = to_dense_dataset([m0])
    assert ds.pose.shape[1:] == (WINDOW, 17, 3)
    assert ds.labels.shape[1] == WINDOW
    assert set(ds.matches.tolist()) == {0}


def _full_person() -> np.ndarray:
    kp = np.zeros((17, 3), np.float32)
    kp[5] = [480, 300, 0.9]
    kp[6] = [520, 300, 0.9]
    kp[11] = [485, 400, 0.9]
    kp[12] = [515, 400, 0.9]
    return kp

from pathlib import Path

import numpy as np
from padel_ml.shot_type_dataset import SEQ_LEN, ShotWindow
from padel_ml.thesis.input_figure import draw_input_figure, step_frames, trail


def test_a_short_window_repeats_frames_and_spans_both_ends() -> None:
    frames = step_frames(10, 20, 100)
    assert len(frames) == 100
    assert frames[0] == 10 and frames[-1] == 20
    assert set(frames.tolist()) == set(range(10, 21))


def test_the_trail_takes_one_step_per_frame_within_reach() -> None:
    frames = np.array([10, 10, 11, 11, 12, 13, 14, 15])
    assert trail(frames, 12, 1) == [2, 4, 5]
    assert trail(frames, 12, 0) == [4]


def test_the_figure_is_written(tmp_path: Path) -> None:
    pose = np.zeros((SEQ_LEN, 17, 3), np.float32)
    pose[:, :, 0] = np.linspace(-1, 1, 17)
    pose[:, :, 1] = np.linspace(-2, 2, 17)
    ball = np.zeros((SEQ_LEN, 3), np.float32)
    ball[:, :] = (0.5, -3.0, 1.0)
    window = ShotWindow(pose, ball, label=2, rally="r", tournament="t", hit_frame=50)
    out = tmp_path / "entrada.pdf"
    draw_input_figure(window, [48, 49, 50, 51, 52], 50, out)
    assert out.stat().st_size > 0

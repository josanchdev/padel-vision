from pathlib import Path

import numpy as np
from padel_ml.thesis.window_figure import (
    EPSILON_S,
    FOREHAND,
    RUN,
    SMASH,
    draw_window_figure,
    lowest,
    racket_tip,
)


def test_epsilon_is_the_systems_margin() -> None:
    assert EPSILON_S == 0.25


def test_the_feet_set_how_far_the_pose_reaches_below_the_hips() -> None:
    assert lowest(SMASH) == 1.3


def test_the_smash_racket_points_up_and_the_forehand_racket_forward() -> None:
    _, up = racket_tip(SMASH)
    _, forward = racket_tip(FOREHAND)
    assert up[1] > 0.9 and forward[0] > 0.9


def test_mirroring_turns_a_pose_around() -> None:
    _, along = racket_tip(RUN)
    _, back = racket_tip(RUN.mirrored())
    assert np.allclose(back, (-along[0], along[1]))


def test_the_figure_is_written(tmp_path: Path) -> None:
    out = tmp_path / "ventana.pdf"
    draw_window_figure(out)
    assert out.stat().st_size > 0

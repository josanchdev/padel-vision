import numpy as np
from padel_ml.thesis.court_marking_figure import (
    CLICKED,
    court_lines_m,
    marking_view,
    to_image,
)

from padel_cv.court_annotator import POINTS_M


def test_court_lines_form_the_t_on_each_side() -> None:
    near, far, centre = court_lines_m()
    assert near == ((0.0, 3.05), (10.0, 3.05))
    assert far[0][1] == far[1][1] == 20.0 - 3.05
    # The centre line runs from one service line to the other, through the net.
    assert centre == ((5.0, 3.05), (5.0, 16.95))


def test_to_image_applies_the_homography() -> None:
    scale_by_ten = np.diag([10.0, 10.0, 1.0])
    assert to_image(scale_by_ten, (2.0, 3.0)) == (20, 30)


def test_marking_view_draws_on_a_copy() -> None:
    # Clicks exactly where a 30 px per metre top view would put the six points.
    points_px = POINTS_M * 30.0 + 50.0
    image = np.zeros((700, 400, 3), np.uint8)
    view = marking_view(image, points_px)

    assert view.shape == image.shape
    assert not image.any()
    x, y = (round(v) for v in points_px[0])
    assert tuple(view[y, x]) == CLICKED

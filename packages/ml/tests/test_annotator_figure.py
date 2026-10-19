import pytest
from padel_ml.thesis.annotator_figure import hit_index

from padel_cv.shot_type_annotator import TypeMark


def test_the_hit_is_found_by_its_frame() -> None:
    marks = [TypeMark(25, "Serve"), TypeMark(58, "Backhand"), TypeMark(99, "Smash")]
    assert hit_index(marks, 99) == 2


def test_a_frame_without_a_labelled_hit_is_refused() -> None:
    with pytest.raises(ValueError):
        hit_index([TypeMark(25, "Serve")], 30)

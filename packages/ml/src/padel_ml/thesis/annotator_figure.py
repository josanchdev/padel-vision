"""The labelling tool figure of section 3.2: what the annotator shows on one hit.

Not a screenshot taken by hand: the frame of the hit, scaled as the tool scales
it, with the tool's own overlay drawn on it (`padel_cv.shot_type_annotator`) and
the labels already saved for that rally. The overlay is the real one: the hit
number and progress at the top left, the label in its colour at the top right,
one dot per hit of the rally, and the keys at the bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2

from padel_cv.arrays import ImageArray
from padel_cv.shot_type_annotator import DISPLAY_WIDTH, TypeMark, _draw_hud, load_marks
from padel_cv.video_io import read_frame


def hit_index(marks: list[TypeMark], hit_frame: int) -> int:
    """Position of the hit at `hit_frame` among the rally's labelled hits."""
    for index, mark in enumerate(marks):
        if mark.frame == hit_frame:
            return index
    raise ValueError(f"no labelled hit at frame {hit_frame}")


def tool_view(video: Path, labels_csv: Path, hit_frame: int) -> ImageArray:
    """The annotator's window at the moment of contact of one labelled hit."""
    saved = load_marks(labels_csv)
    marks = [saved[frame] for frame in sorted(saved)]
    index = hit_index(marks, hit_frame)
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    image = read_frame(video, hit_frame)
    scale = DISPLAY_WIDTH / image.shape[1]
    canvas = cast(ImageArray, cv2.resize(image, (DISPLAY_WIDTH, round(image.shape[0] * scale))))
    _draw_hud(canvas, marks, index, hit_frame, fps, saved=False, is_hit_frame=True)
    return canvas

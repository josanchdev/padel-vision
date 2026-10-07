from pathlib import Path

import cv2
import numpy as np
import pytest
from padel_ml.thesis.problem import Region, draw_problem_figure

from padel_cv.video_io import read_frame


def test_read_frame_returns_exactly_the_frame_asked_for(tmp_path: Path) -> None:
    # Each frame carries its index as its grey level, so a wrong frame shows.
    video = tmp_path / "counter.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 25, (64, 48))
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 8, np.uint8))
    writer.release()

    assert abs(int(read_frame(video, 17).mean()) - 17 * 8) <= 2
    with pytest.raises(ValueError, match="no frame 30"):
        read_frame(video, 30)


def test_the_two_frames_must_be_the_same_size(tmp_path: Path) -> None:
    raw = np.zeros((1080, 1920, 3), np.uint8)
    with pytest.raises(ValueError, match="differ in size"):
        draw_problem_figure(raw, raw[:540], Region(0, 0, 10, 10), tmp_path / "problema.pdf")


def test_region_cuts_the_frame_in_its_own_pixels() -> None:
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    assert Region(x0=2, y0=1, x1=5, y1=4).of(frame).shape == (3, 3, 3)
    assert (Region(x0=2, y0=1, x1=5, y1=4).of(frame)[0, 0] == frame[1, 2]).all()

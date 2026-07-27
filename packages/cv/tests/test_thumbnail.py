import cv2
import numpy as np

from padel_cv.thumbnail import write_thumbnail


def _make_video(path, n_frames, size=(1920, 1080)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


def test_writes_downscaled_thumbnail(tmp_path) -> None:
    video = tmp_path / "match.mp4"
    _make_video(video, n_frames=20)
    out = write_thumbnail(video, tmp_path / "thumb.jpg", width=480)
    assert out is not None
    assert out.exists()
    img = cv2.imread(str(out))
    assert img.shape[1] == 480  # downscaled to requested width
    assert img.shape[0] == 270  # 16:9 preserved


def test_returns_none_for_unreadable_video(tmp_path) -> None:
    assert write_thumbnail(tmp_path / "nope.mp4", tmp_path / "thumb.jpg") is None

from pathlib import Path

import cv2
import numpy as np

from padel_cv.blind_annotator import (
    FrameCache,
    HitMark,
    MarkBook,
    frame_at,
    load_hit_marks,
    save_hit_marks,
    x_of,
)


def _video(path: Path, n_frames: int, size: tuple[int, int] = (1920, 1080)) -> None:
    """Frame i is a flat grey of value 10*i, so a frame can be recognised by colour."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 10 * i, dtype=np.uint8))
    writer.release()


def _grey(image: np.ndarray) -> float:
    return float(image.mean())


def test_marks_roundtrip_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "rally.csv"
    marks = [HitMark(40, 812.5, 603.0, "Smash"), HitMark(12, 100.0, 200.0, "Serve")]
    save_hit_marks(marks, path)
    loaded = load_hit_marks(path)
    assert sorted(loaded) == [12, 40]
    assert loaded[40] == HitMark(40, 812.5, 603.0, "Smash")


def test_every_change_is_on_disk_immediately(tmp_path: Path) -> None:
    """WSL crashes: a mark must survive the process dying right after it."""
    path = tmp_path / "rally.csv"
    MarkBook(path).put(HitMark(10, 1.0, 2.0, "Forehand"))
    assert 10 in load_hit_marks(path)


def test_a_second_mark_on_the_same_frame_replaces_the_first(tmp_path: Path) -> None:
    book = MarkBook(tmp_path / "rally.csv")
    book.put(HitMark(10, 1.0, 2.0, "Forehand"))
    book.put(HitMark(10, 5.0, 6.0, "Backhand"))
    assert len(book.marks) == 1
    assert book.marks[10].shot_type == "Backhand"


def test_undo_reverts_puts_replacements_and_deletes(tmp_path: Path) -> None:
    path = tmp_path / "rally.csv"
    book = MarkBook(path)
    book.put(HitMark(10, 1.0, 2.0, "Forehand"))
    book.put(HitMark(10, 5.0, 6.0, "Backhand"))
    book.delete(10)
    assert book.undo() == 10
    assert book.marks[10].shot_type == "Backhand"  # delete undone
    book.undo()
    assert book.marks[10].shot_type == "Forehand"  # replacement undone
    book.undo()
    assert 10 not in book.marks  # first put undone
    assert load_hit_marks(path) == {}
    assert book.undo() is None


def test_neighbour_jumps_between_marks(tmp_path: Path) -> None:
    book = MarkBook(tmp_path / "rally.csv")
    for frame in (10, 50, 90):
        book.put(HitMark(frame, 0.0, 0.0, "Forehand"))
    assert book.neighbour(50, forward=True) == 90
    assert book.neighbour(50, forward=False) == 10
    assert book.neighbour(90, forward=True) is None


def test_timeline_maps_both_ways() -> None:
    assert frame_at(0, 1280, 1000) == 0
    assert frame_at(1279, 1280, 1000) == 999
    for frame in (0, 123, 999):
        assert abs(frame_at(x_of(frame, 1280, 1000), 1280, 1000) - frame) <= 1


def test_cache_returns_the_exact_frame_forwards_and_backwards(tmp_path: Path) -> None:
    """The label IS the contact frame, so index i must be frame i, in any order."""
    video = tmp_path / "rally.mp4"
    _video(video, 20)
    cache = FrameCache(video)
    for index in (15, 3, 19, 0, 7):
        image = cache.get(index)
        assert image is not None
        assert abs(_grey(image) - 10 * index) < 4, index
    assert cache.get(20) is None
    assert cache.exhausted and cache.n_frames == 20
    cache.release()


def test_clicks_map_back_to_original_pixels(tmp_path: Path) -> None:
    """Shown at 1280 wide, a 1080p click must be stored at 1920-wide coordinates."""
    video = tmp_path / "rally.mp4"
    _video(video, 2)
    cache = FrameCache(video)
    assert cache.display_size == (1280, 720)
    assert cache.to_original == (1.5, 1.5)
    image = cache.get(0)
    assert image is not None and image.shape[:2] == (720, 1280)
    cache.release()

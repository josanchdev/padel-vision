import csv

from padel_cv.shot_type_annotator import (
    SHOT_TYPES,
    TypeMark,
    load_hit_frames,
    load_marks,
    save_marks,
)


def test_hit_frames_come_from_the_gt(tmp_path) -> None:
    csv_path = tmp_path / "hits.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "start", "end", "class", "class_id"])
        w.writerow(["r0.mp4", "1.0", "1.2", "hit", "0"])  # centre 1.1 s -> frame 27.5 -> 28
        w.writerow(["other.mp4", "5.0", "5.2", "hit", "0"])
        w.writerow(["r0.mp4", "2.0", "2.0", "hit", "0"])  # -> frame 50
    assert load_hit_frames(csv_path, "r0.mp4", fps=25.0) == [28, 50]


def test_round_trip_keeps_labels_and_wall_flag(tmp_path) -> None:
    out = tmp_path / "types.csv"
    marks = [
        TypeMark(frame=10, shot_type="Forehand"),
        TypeMark(frame=20, shot_type="Backhand", from_wall=True),
        TypeMark(frame=30),  # still unlabelled
    ]
    save_marks(marks, out)
    back = load_marks(out)
    assert back[10].shot_type == "Forehand"
    assert back[20].from_wall is True
    assert back[30].shot_type is None  # unlabelled hits survive the round trip


def test_four_classes_plus_discard() -> None:
    assert set(SHOT_TYPES.values()) == {"Serve", "Forehand", "Backhand", "Smash", "Other"}


def test_load_marks_on_missing_file(tmp_path) -> None:
    assert load_marks(tmp_path / "nope.csv") == {}


def test_clip_is_centred_on_the_hit() -> None:
    """The loaded clip must put the hit frame where the caller expects it."""
    import cv2

    from padel_cv.shot_type_annotator import _load_clip

    video = "data/raw/padel_audio_dataset/CVSPORTS_Padel/rallies/20230528_VIGO_00.mp4"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        import pytest

        pytest.skip("dataset video not available")
    clip, hit_at = _load_clip(cap, hit_frame=120, half=25)
    cap.release()
    assert len(clip) == 51  # 25 either side + the hit itself
    assert hit_at == 25  # the hit sits in the middle


def test_clip_near_the_start_is_clamped() -> None:
    import cv2

    from padel_cv.shot_type_annotator import _load_clip

    video = "data/raw/padel_audio_dataset/CVSPORTS_Padel/rallies/20230528_VIGO_00.mp4"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        import pytest

        pytest.skip("dataset video not available")
    clip, hit_at = _load_clip(cap, hit_frame=10, half=25)
    cap.release()
    assert hit_at == 10  # cannot go before frame 0, so the hit shifts
    assert len(clip) > 0

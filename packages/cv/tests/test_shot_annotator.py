import json

from padel_cv.shot_annotator import (
    ShotMark,
    _load_ball,
    _marks_summary,
    load_marks,
    save_marks,
)


def test_save_and_load_roundtrip(tmp_path) -> None:
    marks = [
        ShotMark(frame=100, type="Forehand", from_wall=False, player=-1),
        ShotMark(frame=50, type="Backhand", from_wall=True, player=2),
    ]
    csv_path = tmp_path / "shots.csv"
    save_marks(marks, csv_path)
    loaded = load_marks(csv_path)
    # saved sorted by frame
    assert [m.frame for m in loaded] == [50, 100]
    assert loaded[0].type == "Backhand" and loaded[0].from_wall is True
    assert loaded[0].player == 2
    assert loaded[1].from_wall is False


def test_load_marks_missing_file_is_empty(tmp_path) -> None:
    assert load_marks(tmp_path / "nope.csv") == []


def test_marks_summary_counts_types() -> None:
    marks = [
        ShotMark(1, "Forehand", False, -1),
        ShotMark(2, "Forehand", False, -1),
        ShotMark(3, "Smash", True, -1),
    ]
    assert _marks_summary(marks) == {"Forehand": 2, "Smash": 1}


def test_load_ball_simple_list(tmp_path) -> None:
    p = tmp_path / "ball.json"
    p.write_text(json.dumps([{"frame": 5, "x": 100.0, "y": 200.0}]))
    ball = _load_ball(p)
    assert ball[5] == (100.0, 200.0)


def test_load_ball_coco(tmp_path) -> None:
    p = tmp_path / "ball_coco.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame_000007.PNG"}],
                "annotations": [{"image_id": 1, "bbox": [10, 20, 4, 4]}],
            }
        )
    )
    ball = _load_ball(p)
    assert ball[7] == (12.0, 22.0)  # bbox centre


def test_load_ball_none_is_empty() -> None:
    assert _load_ball(None) == {}


def test_resume_frame_prefers_saved_position(tmp_path) -> None:
    from padel_cv.shot_annotator import _load_resume_frame, _save_pos

    csv_path = tmp_path / "shots.csv"
    marks = [ShotMark(100, "Forehand", False, -1)]
    # No sidecar yet -> resume just after the last mark.
    assert _load_resume_frame(csv_path, marks) == 100
    # Saved position wins.
    _save_pos(csv_path, 1500)
    assert _load_resume_frame(csv_path, marks) == 1500


def test_resume_frame_empty_is_zero(tmp_path) -> None:
    from padel_cv.shot_annotator import _load_resume_frame

    assert _load_resume_frame(tmp_path / "shots.csv", []) == 0


def test_ball_corrections_roundtrip(tmp_path) -> None:
    from padel_cv.shot_annotator import load_ball_corrections, save_ball_corrections

    csv_path = tmp_path / "shots.csv"
    corr = {100: (640.0, 360.0), 250: (12.5, 88.0)}
    save_ball_corrections(csv_path, corr)
    loaded = load_ball_corrections(csv_path)
    assert loaded == corr
    assert load_ball_corrections(tmp_path / "nope.csv") == {}

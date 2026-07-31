import csv
import json

from padel_ml.shot_eval import (
    ShotBlock,
    evaluate_detection,
    load_ball_track,
    load_shot_blocks,
)

from padel_cv.ball_events import BallSample


def _write_shots_csv(path, flags_and_cats) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["file_name", "has_shot", "category"])
        for i, (has, cat) in enumerate(flags_and_cats):
            w.writerow([f"frame_{i:06d}.PNG", has, cat])


def test_shot_blocks_collapse_contiguous_runs(tmp_path) -> None:
    csv_path = tmp_path / "s.csv"
    # two shots: frames 1-3 (Forehand) and 6-7 (Smash), 0 elsewhere
    rows = [(0, 0)] + [(1, "Forehand")] * 3 + [(0, 0)] * 2 + [(1, "Smash")] * 2 + [(0, 0)]
    _write_shots_csv(csv_path, rows)
    blocks = load_shot_blocks(csv_path)
    assert blocks == [ShotBlock(1, 3, "Forehand"), ShotBlock(6, 7, "Smash")]
    assert blocks[0].centre == 2


def test_ball_track_dedups_multiple_boxes_per_frame(tmp_path) -> None:
    data = {
        "images": [
            {"id": 1, "file_name": "frame_000000.PNG"},
            {"id": 2, "file_name": "frame_000001.PNG"},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [10, 10, 2, 2]},
            {"image_id": 1, "bbox": [500, 500, 2, 2]},  # duplicate on same frame
            {"image_id": 2, "bbox": [20, 20, 4, 4]},
        ],
    }
    p = tmp_path / "ball.json"
    p.write_text(json.dumps(data))
    track = load_ball_track(p)
    assert [s.frame_index for s in track] == [0, 1]  # one sample per frame
    assert track[0] == BallSample(0, 11.0, 11.0)


def test_detection_matches_shot_near_block_centre() -> None:
    # A ball flying in then sharply reversing at frame 10, next to a wrist.
    ball = [BallSample(i, 100 + 15 * i, 100 - 15 * i) for i in range(11)]
    ball += [BallSample(10 + i, 250 - 15 * i, -50 + 15 * i) for i in range(1, 8)]
    # A wrist tracks the ball around the reversal (the detected turn may land a
    # frame either side of 10), so it reads as a shot rather than a bounce.
    ball_xy = {s.frame_index: (s.x_px, s.y_px) for s in ball}
    wrists = {f: [([ball_xy[f]], 100.0)] for f in (9, 10, 11)}
    blocks = [ShotBlock(7, 13, "Forehand")]  # centre 10
    r = evaluate_detection(ball, wrists, blocks, tol=3)
    assert r.covered == 1
    assert r.recall == 1.0


def test_detection_misses_when_no_reversal() -> None:
    ball = [BallSample(i, 100 + 10 * i, 100) for i in range(20)]  # straight
    wrists = {10: [([(200.0, 100.0)], 100.0)]}
    blocks = [ShotBlock(8, 12, "Forehand")]
    r = evaluate_detection(ball, wrists, blocks, tol=3)
    assert r.covered == 0
    assert r.recall == 0.0


def test_load_marked_csv_format(tmp_path) -> None:
    """The quick-mark annotator CSV (one row per shot) loads as centred blocks."""
    from padel_ml.shot_eval import load_shot_blocks

    p = tmp_path / "marks.csv"
    p.write_text("frame;type;from_wall;player\n100;Forehand;0;-1\n250;Backhand;1;2\n")
    blocks = load_shot_blocks(p)
    assert len(blocks) == 2
    assert blocks[0].centre == 100 and blocks[0].category == "Forehand"
    assert blocks[1].centre == 250 and blocks[1].category == "Backhand"

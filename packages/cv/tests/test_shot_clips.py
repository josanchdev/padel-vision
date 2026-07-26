import csv
import json
from pathlib import Path

import numpy as np

from padel_cv.shot_clips import (
    ShotRun,
    attribute_shot,
    canonical_label,
    load_ball_positions,
    load_shot_runs,
)


def skeleton_with_wrist(x: float, y: float) -> np.ndarray:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[10] = [x, y, 0.9]  # right wrist
    return kp


def test_canonical_label_folds_dropshot() -> None:
    assert canonical_label("Dropshot") == "Other"
    assert canonical_label("Forehand") == "Forehand"


def test_load_shot_runs_groups_contiguous(tmp_path: Path) -> None:
    csv_path = tmp_path / "shots.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["file_name", "has_shot", "category"])
        for i in range(6):
            has = "1" if i in (1, 2, 4) else "0"
            cat = "Smash" if i in (1, 2) else "Forehand" if i == 4 else "0"
            w.writerow([f"frame_{i:06d}.PNG", has, cat])
    runs = load_shot_runs(csv_path)
    assert runs == [ShotRun(1, 2, "Smash"), ShotRun(4, 4, "Forehand")]


def test_load_ball_positions_uses_bbox_center(tmp_path: Path) -> None:
    ball = {
        "images": [{"id": 1, "file_name": "frame_000003.PNG"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 4.0, 6.0]}],
        "categories": [{"id": 1, "name": "Ball"}],
    }
    path = tmp_path / "ball.json"
    path.write_text(json.dumps(ball))
    assert load_ball_positions(path) == {3: (12.0, 23.0)}


def test_attribute_shot_picks_nearest_wrist_and_impact_frame() -> None:
    run = ShotRun(0, 2, "Smash")
    # Two players; the ball nears player 1's wrist only at frame 1.
    poses = {
        0: np.stack([skeleton_with_wrist(0, 0), skeleton_with_wrist(500, 500)]),
        1: np.stack([skeleton_with_wrist(0, 0), skeleton_with_wrist(105, 100)]),
        2: np.stack([skeleton_with_wrist(0, 0), skeleton_with_wrist(500, 500)]),
    }
    ball = {0: (300.0, 300.0), 1: (100.0, 100.0), 2: (300.0, 300.0)}
    shot = attribute_shot(run, ball, poses)
    assert shot is not None
    assert shot.impact_frame == 1
    assert shot.player_index == 1
    assert shot.wrist_ball_dist_px < 10


def test_attribute_shot_none_without_ball_or_poses() -> None:
    run = ShotRun(0, 1, "Serve")
    assert attribute_shot(run, {}, {0: np.zeros((1, 17, 3), np.float32)}) is None
    assert attribute_shot(run, {0: (1.0, 1.0)}, {}) is None

import json
from pathlib import Path

import numpy as np
from padel_ml.rally_analysis import RallyAnalysis, Shot
from padel_ml.web_export import SCHEMA_VERSION, PointMeta, describe, point_record, write_index

from padel_cv.pipeline import PoseDetection


def test_cvsports_names_become_readable_titles() -> None:
    meta = describe("20230528_VIGO_01")
    assert meta == PointMeta("Punto 01 · Vigo", "Vigo", "2023-05-28")
    assert describe("20231008_DUITSLAND_00").city == "Alemania"


def test_other_videos_keep_their_name_unless_titled() -> None:
    assert describe("miami_rally1") == PointMeta("miami_rally1", None, None)
    assert describe("miami_rally1", "Final Miami").title == "Final Miami"


def _pose(player_id: int, x: float, y: float) -> PoseDetection:
    return PoseDetection(
        bbox_xyxy=(0.0, 0.0, 10.0, 20.0),
        confidence=1.0,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        player_id=player_id,
        court_position_m=(x, y),
    )


def test_point_record_follows_the_contract() -> None:
    analysis = RallyAnalysis(
        video=Path("20230528_VIGO_01.mp4"),
        fps=25.0,
        width=1920,
        height=1080,
        n_frames=50,
        court=None,
        shots=[Shot(10, 0.4, 3, "Serve", 0.91234, None), Shot(30, 1.2, None, None, 0.0, None)],
        players={0: [_pose(1, 2.345, 15.678), _pose(3, 4.0, 4.0)], 1: [_pose(1, 2.4, 15.7)]},
        ball={},
        ball_smoothed={},
        bounces=[],
    )
    record = point_record("20230528_VIGO_01", describe("20230528_VIGO_01"), analysis)
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["duration_s"] == 2.0
    assert record["shots"][0] == {
        "frame": 10,
        "t": 0.4,
        "player": 3,
        "type": "Serve",
        "confidence": 0.912,
    }
    assert record["shots"][1]["player"] is None and record["shots"][1]["type"] is None
    assert record["players"]["1"] == [[0, 2.35, 15.68], [1, 2.4, 15.7]]
    assert record["players"]["3"] == [[0, 4.0, 4.0]]
    json.dumps(record)  # plain JSON, nothing numpy left in it


def test_index_lists_every_point_in_date_order(tmp_path: Path) -> None:
    for point_id, day in (("b", "2023-06-04"), ("a", "2023-05-28")):
        (tmp_path / point_id).mkdir()
        (tmp_path / point_id / "point.json").write_text(
            json.dumps(
                {
                    "id": point_id,
                    "title": point_id,
                    "city": "X",
                    "date": day,
                    "duration_s": 5.0,
                    "shots": [{}, {}],
                }
            )
        )
    index = json.loads(write_index(tmp_path).read_text())
    assert [p["id"] for p in index["points"]] == ["a", "b"]
    assert index["points"][0]["shots"] == 2

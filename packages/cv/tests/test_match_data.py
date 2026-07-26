import csv
import json

import numpy as np

from padel_cv.bounces import Bounce
from padel_cv.match_data import SCHEMA_VERSION, MatchAnalysis
from padel_cv.pipeline import BallDetection, Frame, PoseDetection, ShotEvent


def _pose(player_id, track_id=7, court=(2.0, 5.0), on_court=True) -> PoseDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    p = PoseDetection((0, 0, 10, 20), 0.9, kp, track_id=track_id)
    p.player_id = player_id
    p.court_position_m = court
    p.on_court = on_court
    return p


def _frame(index, ball=None) -> Frame:
    f = Frame(index=index, timestamp_s=index / 30.0, image=np.zeros((2, 2, 3), np.uint8))
    f.ball = ball
    return f


def test_accumulate_players_shots_ball() -> None:
    analysis = MatchAnalysis(source_video="m.mp4", fps=30.0)
    ball = BallDetection(image_xy=(100.0, 200.0), confidence=0.9, court_xy_m=(3.0, 4.0))
    frame = _frame(5, ball=ball)
    frame.poses = [_pose(1), _pose(2)]
    frame.shot_events = [ShotEvent(player_id=1, frame_index=5, label="Forehand", confidence=0.8)]
    analysis.accumulate_frame(frame)

    assert len(analysis.players) == 2
    assert analysis.players[0].player_id == 1
    assert analysis.players[0].court_x_m == 2.0
    assert len(analysis.shots) == 1
    assert analysis.shots[0].label == "Forehand"
    assert len(analysis.ball) == 1
    assert analysis.ball[0].court_x_m == 3.0


def test_unidentified_players_are_skipped() -> None:
    analysis = MatchAnalysis(source_video="m.mp4", fps=30.0)
    frame = _frame(0)
    frame.poses = [_pose(None)]  # no player_id -> not recorded
    analysis.accumulate_frame(frame)
    assert analysis.players == []


def test_json_roundtrip_has_schema_version(tmp_path) -> None:
    analysis = MatchAnalysis(source_video="m.mp4", fps=30.0)
    frame = _frame(0)
    frame.poses = [_pose(1)]
    analysis.accumulate_frame(frame)
    analysis.set_bounces([Bounce(frame_index=10, x_px=50.0, y_px=60.0)])

    out = tmp_path / "a.json"
    analysis.write_json(out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["source_video"] == "m.mp4"
    assert len(data["players"]) == 1
    assert data["bounces"][0]["frame_index"] == 10


def test_csv_export_one_row_per_event(tmp_path) -> None:
    analysis = MatchAnalysis(source_video="m.mp4", fps=30.0)
    frame = _frame(3, ball=BallDetection(image_xy=(1.0, 2.0), confidence=0.7))
    frame.poses = [_pose(1)]
    frame.shot_events = [ShotEvent(player_id=1, frame_index=3, label="Smash", confidence=0.9)]
    analysis.accumulate_frame(frame)
    analysis.set_bounces([Bounce(frame_index=3, x_px=5.0, y_px=6.0)])

    out = tmp_path / "a.csv"
    analysis.write_csv(out)
    rows = list(csv.DictReader(out.open()))
    types = [r["event_type"] for r in rows]
    assert types == ["player", "shot", "ball", "bounce"]
    shot_row = next(r for r in rows if r["event_type"] == "shot")
    assert shot_row["label"] == "Smash"

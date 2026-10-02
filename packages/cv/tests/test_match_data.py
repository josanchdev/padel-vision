import json

from padel_cv.match_data import (
    SCHEMA_VERSION,
    BounceRecord,
    MatchAnalysis,
    PlayerFrameRecord,
    ShotRecord,
)


def test_json_roundtrip_has_schema_version(tmp_path) -> None:
    analysis = MatchAnalysis(
        source_video="m.mp4",
        fps=30.0,
        players=[
            PlayerFrameRecord(
                frame_index=0,
                timestamp_s=0.0,
                player_id=1,
                track_id=7,
                court_x_m=2.0,
                court_y_m=5.0,
                on_court=True,
            )
        ],
        shots=[
            ShotRecord(frame_index=3, timestamp_s=0.1, player_id=1, label="Smash", confidence=0.9)
        ],
        bounces=[BounceRecord(frame_index=10, image_x=50.0, image_y=60.0)],
    )

    out = tmp_path / "a.json"
    analysis.write_json(out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["source_video"] == "m.mp4"
    assert data["players"][0]["court_x_m"] == 2.0
    assert data["shots"][0]["label"] == "Smash"
    assert data["bounces"][0]["frame_index"] == 10

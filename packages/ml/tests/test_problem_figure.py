from pathlib import Path

import numpy as np
from padel_ml.thesis.problem import Crop, Hit, spotlight, system_hits

from padel_cv.detections import PoseDetection

HEADER = "rally;tournament;truth_s;detected_s;truth_type;truth_player;player;shot_type\n"


def test_system_hits_keeps_only_what_the_system_returned_for_that_rally(tmp_path: Path) -> None:
    csv = tmp_path / "hits.csv"
    csv.write_text(
        HEADER
        + "R_01;R;2.7;2.73;Smash;2;2;Smash\n"
        + "R_01;R;0.3;0.30;Serve;1;1;Serve\n"
        + "R_01;R;5.0;;Forehand;4;;\n"  # missed by the audio: nothing to draw
        + "R_02;R;1.0;1.01;Backhand;3;3;Backhand\n"
    )
    assert system_hits("R_01", csv) == [Hit(0.30, 1, "Serve"), Hit(2.73, 2, "Smash")]


def test_crop_moves_a_player_box_into_the_shown_region() -> None:
    crop = Crop(x0=100, y0=50, x1=1100, y1=550)
    player = PoseDetection(
        bbox_xyxy=(300.0, 150.0, 400.0, 350.0),
        confidence=1.0,
        keypoints=np.zeros((17, 3), np.float32),
        player_id=3,
    )
    assert crop.aspect == 2.0
    assert crop.box(player) == (200.0, 100.0, 300.0, 300.0)


def test_spotlight_keeps_the_hitter_and_dims_the_rest() -> None:
    image = np.full((200, 400, 3), 200, np.uint8)
    lit = spotlight(image, (180.0, 80.0, 220.0, 160.0))
    assert lit[120, 200, 0] > 190  # the hitter keeps his brightness
    assert lit[10, 10, 0] < 120  # a corner far from him is dimmed

from pathlib import Path

import cv2
import numpy as np
from padel_ml.external_eval import clicked_player, frozen_frames, score_clip, summarize
from padel_ml.rally_analysis import RallyAnalysis, Shot

from padel_cv.blind_annotator import HitMark
from padel_cv.pipeline import PoseDetection

FPS = 25.0


def _pose(player_id: int, cx: float, cy: float, court_y_m: float) -> PoseDetection:
    """A player box 200 px tall centred on (cx, cy), standing at court_y_m."""
    return PoseDetection(
        bbox_xyxy=(cx - 50, cy - 100, cx + 50, cy + 100),
        confidence=1.0,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        player_id=player_id,
        court_position_m=(5.0, court_y_m),
    )


def _analysis(shots: list[Shot], n_frames: int = 300) -> RallyAnalysis:
    """J1 near side (left), J2 near side (right), J3 far side: fixed all rally."""
    players = {
        i: [_pose(1, 400, 800, 15.0), _pose(2, 1400, 800, 15.0), _pose(3, 900, 300, 4.0)]
        for i in range(n_frames)
    }
    return RallyAnalysis(
        video=Path("clip.mp4"),
        fps=FPS,
        width=1920,
        height=1080,
        n_frames=n_frames,
        court=None,
        shots=shots,
        players=players,
        ball={},
        ball_smoothed={},
        bounces=[],
    )


def _shot(frame: int, player: int | None, shot_type: str | None) -> Shot:
    return Shot(frame, frame / FPS, player, shot_type, 0.9, None)


def test_click_resolves_to_the_player_standing_there() -> None:
    analysis = _analysis([])
    pose = clicked_player(HitMark(50, 1410.0, 790.0, "Forehand"), analysis.players)
    assert pose is not None and pose.player_id == 2


def test_click_on_nobody_means_hitter_not_detected() -> None:
    analysis = _analysis([])
    assert clicked_player(HitMark(50, 10.0, 10.0, "Other"), analysis.players) is None


def test_detection_pairs_within_the_collar_only() -> None:
    """6 frames = 240 ms pairs; 10 frames = 400 ms is a miss plus a false alarm."""
    truth = [HitMark(100, 400.0, 800.0, "Forehand"), HitMark(200, 900.0, 300.0, "Smash")]
    analysis = _analysis([_shot(106, 1, "Forehand"), _shot(210, 3, "Smash")])
    numbers = summarize(score_clip("c", truth, analysis))
    assert numbers["detection_precision"] == 0.5
    assert numbers["detection_recall"] == 0.5


def test_player_team_and_type_are_judged_separately() -> None:
    truth = [
        HitMark(50, 400.0, 800.0, "Forehand"),  # J1 hit it
        HitMark(100, 400.0, 800.0, "Backhand"),  # J1 hit it
        HitMark(150, 900.0, 300.0, "Smash"),  # J3 hit it
    ]
    analysis = _analysis(
        [
            _shot(50, 1, "Forehand"),  # all right
            _shot(100, 2, "Backhand"),  # wrong player, right team, right type
            _shot(150, 1, "Forehand"),  # wrong player, wrong team, wrong type
        ]
    )
    rows = score_clip("c", truth, analysis)
    assert [r.player_ok for r in rows] == [True, False, False]
    assert [r.team_ok for r in rows] == [True, True, False]
    numbers = summarize(rows)
    assert abs(numbers["type_accuracy"] - 2 / 3) < 1e-3
    assert numbers["type_accuracy_right_player"] == 1.0  # the classifier alone
    assert abs(numbers["end_to_end"] - 1 / 3) < 1e-3


def test_other_counts_for_detection_but_not_for_type() -> None:
    truth = [HitMark(50, 400.0, 800.0, "Other"), HitMark(150, 900.0, 300.0, "Smash")]
    analysis = _analysis([_shot(50, 1, "Forehand"), _shot(150, 3, "Smash")])
    rows = score_clip("c", truth, analysis)
    assert rows[0].type_ok is None
    numbers = summarize(rows)
    assert numbers["detection_f1"] == 1.0
    assert numbers["type_accuracy"] == 1.0


def test_unclassified_hit_counts_as_a_type_error() -> None:
    truth = [HitMark(50, 400.0, 800.0, "Forehand")]
    rows = score_clip("c", truth, _analysis([_shot(50, 1, None)]))
    assert rows[0].type_ok is False


def test_hits_labelled_inside_a_freeze_are_flagged() -> None:
    truth = [HitMark(50, 400.0, 800.0, "Forehand")]
    rows = score_clip("c", truth, _analysis([_shot(50, 1, "Forehand")]), frozen={48, 49, 50, 51})
    assert rows[0].in_freeze


def test_freeze_detector_finds_a_frozen_run_but_not_single_repeats(tmp_path: Path) -> None:
    """Frames 10-17 repeat one image (a broadcast freeze); frame 5 repeats once
    (the 25->30 fps duplicate), which must not count."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 360))
    images = []
    for i in range(25):
        source = 10 if 10 <= i <= 17 else (4 if i == 5 else i)
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.circle(image, (40 + 20 * source, 180), 30, (255, 255, 255), -1)  # a moving ball
        images.append(image)
    for image in images:
        writer.write(image)
    writer.release()
    frozen = frozen_frames(path)
    assert set(range(10, 18)) <= frozen
    assert 5 not in frozen and 4 not in frozen

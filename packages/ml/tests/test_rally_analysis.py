from pathlib import Path

import numpy as np
from padel_ml.rally_analysis import UNKNOWN_PLAYER, RallyAnalysis, Shot, resolve_shots
from padel_ml.shot_type_dataset import CLASSES
from padel_ml.shot_type_train import best_class

from padel_cv.pipeline import PoseDetection

FPS = 25.0
SERVE = CLASSES.index("Serve")
SMASH = CLASSES.index("Smash")


def _player(player_id: int, cx: float, cy: float) -> PoseDetection:
    """A standing player centred at (cx, cy): hips at cy, wrists out to the sides."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[:5, :2] = (cx, cy - 90)  # head
    kp[5], kp[6] = (cx - 20, cy - 60, 1), (cx + 20, cy - 60, 1)  # shoulders
    kp[7], kp[8] = (cx - 30, cy - 50, 1), (cx + 30, cy - 50, 1)  # elbows
    kp[9], kp[10] = (cx - 40, cy - 40, 1), (cx + 40, cy - 40, 1)  # wrists
    kp[11], kp[12] = (cx - 15, cy, 1), (cx + 15, cy, 1)  # hips
    kp[13], kp[14] = (cx - 15, cy + 45, 1), (cx + 15, cy + 45, 1)  # knees
    kp[15], kp[16] = (cx - 15, cy + 90, 1), (cx + 15, cy + 90, 1)  # ankles
    return PoseDetection(
        bbox_xyxy=(cx - 50, cy - 100, cx + 50, cy + 100),
        confidence=1.0,
        keypoints=kp,
        player_id=player_id,
    )


Players = dict[int, list[PoseDetection]]
Ball = dict[int, tuple[float, float]]


def _rally(n_frames: int = 200) -> tuple[Players, Ball]:
    """J1 on the left, J3 on the right; the ball flies between their right wrists.

    It sits on J1's wrist at frame 50 and on J3's at frame 120.
    """
    players = {i: [_player(1, 300, 600), _player(3, 1500, 300)] for i in range(n_frames)}
    j1_wrist, j3_wrist = np.array([340.0, 560.0]), np.array([1540.0, 260.0])
    ball: Ball = {}
    for i in range(n_frames):
        t = float(np.clip((i - 50) / 70, 0.0, 1.0))
        x, y = (1 - t) * j1_wrist + t * j3_wrist
        ball[i] = (float(x), float(y))
    return players, ball


def _always(probabilities: list[float]):  # type: ignore[no-untyped-def]
    return lambda _window: np.array(probabilities, dtype=np.float32)


def test_best_class_is_plain_argmax_when_a_serve_is_allowed() -> None:
    assert best_class([0.1, 0.1, 0.2, 0.6], may_serve=True) == SERVE


def test_best_class_never_gives_a_serve_mid_rally() -> None:
    """A serve ranked first mid-rally falls to the next-best class."""
    assert best_class([0.1, 0.1, 0.2, 0.6], may_serve=False) == SMASH


def test_best_class_leaves_other_classes_untouched() -> None:
    assert best_class([0.7, 0.1, 0.1, 0.1], may_serve=False) == 0


def test_resolve_shots_assigns_each_hit_to_the_player_holding_the_ball() -> None:
    players, ball = _rally()
    shots = resolve_shots([50, 120], players, ball, FPS, _always([0.1, 0.1, 0.2, 0.6]))
    assert [s.player_id for s in shots] == [1, 3]


def test_only_the_opening_hit_may_be_a_serve() -> None:
    """Same classifier output for both hits: the first keeps the serve, the second cannot."""
    players, ball = _rally()
    shots = resolve_shots([50, 120], players, ball, FPS, _always([0.1, 0.1, 0.2, 0.6]))
    assert [s.shot_type for s in shots] == ["Serve", "Smash"]
    assert shots[1].confidence == np.float32(0.2)
    # the raw output is kept as the model gave it, before the rule
    assert shots[1].probabilities is not None
    assert int(np.argmax(shots[1].probabilities)) == SERVE


def test_below_threshold_is_left_unclassified_but_keeps_its_confidence() -> None:
    players, ball = _rally()
    shots = resolve_shots(
        [50, 120], players, ball, FPS, _always([0.1, 0.1, 0.2, 0.6]), min_confidence=0.7
    )
    assert shots[0].shot_type is None
    assert shots[0].player_id == 1
    assert abs(shots[0].confidence - 0.6) < 1e-6


def test_hit_with_nobody_around_is_reported_not_dropped() -> None:
    shots = resolve_shots([50], {}, {}, FPS, _always([1.0, 0.0, 0.0, 0.0]))
    assert len(shots) == 1
    assert shots[0].player_id is None
    assert shots[0].shot_type is None
    assert shots[0].probabilities is None


def test_match_data_marks_unknown_players_and_unclassified_shots() -> None:
    pose = _player(2, 500, 500)
    pose.court_position_m = (4.0, 12.0)
    pose.on_court = True
    analysis = RallyAnalysis(
        video=Path("rally.mp4"),
        fps=FPS,
        width=1920,
        height=1080,
        n_frames=1,
        court=None,
        shots=[
            Shot(0, 0.0, None, None, 0.0, None),
            Shot(10, 0.4, 2, "Forehand", 0.9, (0.9, 0.05, 0.03, 0.02)),
        ],
        players={0: [pose]},
        ball={},
        ball_smoothed={},
        bounces=[],
    )
    data = analysis.to_match_data().to_dict()
    assert [s["player_id"] for s in data["shots"]] == [UNKNOWN_PLAYER, 2]
    assert [s["label"] for s in data["shots"]] == ["Unclassified", "Forehand"]
    assert data["players"][0]["court_x_m"] == 4.0
    assert data["players"][0]["court_y_m"] == 12.0

from padel_ml.ball_infer import BallHit
from padel_ml.ball_postprocess import interpolate_gaps, postprocess_ball, remove_teleports


def _h(i, x, y, c=0.9):
    return BallHit(frame_index=i, x_px=float(x), y_px=float(y), confidence=c)


def test_removes_a_teleporting_detection() -> None:
    # smooth trajectory with one frame jumping across the picture and back
    hits = [_h(0, 100, 100), _h(1, 120, 105), _h(2, 1800, 900), _h(3, 160, 115), _h(4, 180, 120)]
    kept = remove_teleports(hits)
    assert [h.frame_index for h in kept] == [0, 1, 3, 4]


def test_keeps_a_genuine_fast_trajectory() -> None:
    hits = [_h(i, 100 + 60 * i, 100 + 20 * i) for i in range(6)]
    assert len(remove_teleports(hits)) == 6


def test_interpolates_short_gaps_only() -> None:
    hits = [_h(0, 100, 100), _h(3, 160, 130), _h(40, 900, 500)]
    out = interpolate_gaps(hits, max_gap_frames=6)
    idx = [h.frame_index for h in out]
    assert idx == [0, 1, 2, 3, 40]  # 4..39 is too long a gap to invent
    mid = next(h for h in out if h.frame_index == 2)
    assert mid.confidence == 0.0  # marked as interpolated
    assert 130 < mid.x_px < 150


def test_postprocess_runs_both_steps() -> None:
    hits = [_h(0, 100, 100), _h(1, 120, 105), _h(2, 1800, 900), _h(4, 160, 115)]
    out = postprocess_ball(hits)
    assert [h.frame_index for h in out] == [0, 1, 2, 3, 4]  # teleport dropped, gap filled

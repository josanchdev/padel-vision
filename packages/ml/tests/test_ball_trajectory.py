import numpy as np
from padel_ml.ball_infer import BallHit
from padel_ml.ball_trajectory import clean_track, reject_outliers, smooth_and_fill


def _parabola(n=14, x0=100.0, vx=25.0, y0=400.0, vy=-40.0, g=4.0):
    """A ball flying like a ball: constant x speed, gravity in y."""
    return [BallHit(i, x0 + vx * i, y0 + vy * i + 0.5 * g * i * i, 0.9) for i in range(n)]


def test_clean_trajectory_survives_untouched() -> None:
    """The filter must not punish a ball that is behaving."""
    hits = _parabola()
    assert len(reject_outliers(hits)) == len(hits)


def test_outlier_off_the_curve_is_dropped() -> None:
    hits = _parabola()
    hits[7] = BallHit(7, 1500.0, 90.0, 0.9)  # detector latched onto something else
    kept = reject_outliers(hits)
    assert 7 not in [h.frame_index for h in kept]
    assert len(kept) == len(hits) - 1


def test_gap_is_filled_along_the_curve() -> None:
    """A filled point must land on the parabola, not on a straight line."""
    hits = [h for h in _parabola() if h.frame_index not in (6, 7)]
    track = smooth_and_fill(hits)
    filled = {p.frame_index: p for p in track if not p.measured}
    assert 6 in filled and 7 in filled
    expected_y = 400.0 - 40.0 * 6 + 0.5 * 4.0 * 36
    assert abs(filled[6].y_px - expected_y) < 5.0


def test_filled_points_are_flagged() -> None:
    """Downstream code must be able to tell a guess from an observation."""
    hits = [h for h in _parabola() if h.frame_index != 5]
    track = smooth_and_fill(hits)
    assert [p.measured for p in track if p.frame_index == 5] == [False]
    assert all(p.measured for p in track if p.frame_index in (0, 1, 2))


def test_long_gaps_are_not_invented() -> None:
    hits = _parabola(6) + [BallHit(i, 900.0 + i, 300.0, 0.9) for i in range(40, 46)]
    frames = {p.frame_index for p in smooth_and_fill(hits)}
    assert not (set(range(12, 38)) & frames)  # the 30-frame hole stays empty


def test_smoothing_reduces_jitter() -> None:
    rng = np.random.default_rng(0)
    hits = [
        BallHit(h.frame_index, h.x_px + rng.normal(0, 6), h.y_px + rng.normal(0, 6), 0.9)
        for h in _parabola(16)
    ]

    def jerk(xs, ys):
        speeds = [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(len(xs) - 1)]
        return float(np.mean(np.abs(np.diff(speeds))))

    before = jerk([h.x_px for h in hits], [h.y_px for h in hits])
    track = clean_track(hits)
    after = jerk([p.x_px for p in track], [p.y_px for p in track])
    assert after < before / 2

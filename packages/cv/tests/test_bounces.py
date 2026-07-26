from padel_cv.bounces import BallSample, detect_bounces


def _arc(frames_ys, x=100.0):
    """Build samples from a list of (frame, y) with constant x."""
    return [BallSample(f, x, y) for f, y in frames_ys]


def test_detects_single_bounce_at_valley() -> None:
    # Ball descends (y grows) to a bottom at frame 4, then rises.
    samples = _arc([(0, 10), (1, 30), (2, 50), (3, 70), (4, 90), (5, 70), (6, 50), (7, 30)])
    bounces = detect_bounces(samples, shot_frames=set())
    assert len(bounces) == 1
    assert bounces[0].frame_index == 4
    assert bounces[0].y_px == 90


def test_shot_coincident_valley_is_discarded() -> None:
    # Same valley, but a player hit at frame 4 -> not a floor bounce.
    samples = _arc([(0, 10), (1, 30), (2, 50), (3, 70), (4, 90), (5, 70), (6, 50), (7, 30)])
    bounces = detect_bounces(samples, shot_frames={4}, shot_guard=2)
    assert bounces == []


def test_jitter_below_prominence_is_ignored() -> None:
    # Near-flat path with a 1px wobble: prominence too small to be a bounce.
    samples = _arc([(0, 50), (1, 50), (2, 51), (3, 50), (4, 50), (5, 50), (6, 50)])
    bounces = detect_bounces(samples, shot_frames=set(), min_prominence_px=4.0)
    assert bounces == []


def test_two_bounces_in_a_rally() -> None:
    # Down-up-down-up: two arc bottoms at frames 3 and 9.
    ys = [10, 40, 70, 95, 70, 40, 20, 45, 75, 98, 75, 45, 20]
    samples = _arc([(i, y) for i, y in enumerate(ys)])
    bounces = detect_bounces(samples, shot_frames=set(), window=2)
    frames = [b.frame_index for b in bounces]
    assert frames == [3, 9]


def test_bounce_near_shot_but_outside_guard_survives() -> None:
    samples = _arc([(0, 10), (1, 30), (2, 50), (3, 70), (4, 90), (5, 70), (6, 50), (7, 30)])
    # Shot at frame 0, guard 2 -> valley at 4 is far enough, survives.
    bounces = detect_bounces(samples, shot_frames={0}, shot_guard=2)
    assert len(bounces) == 1
    assert bounces[0].frame_index == 4

from padel_cv.ball_events import (
    BallSample,
    classify_events,
    detect_direction_changes,
)


def _straight(n=12):
    # Ball moving steadily right — no direction change.
    return [BallSample(i, 10.0 * i, 50.0) for i in range(n)]


def _v_shape():
    # Ball goes right-down, then sharply right-up: a clear direction change at 5.
    down = [BallSample(i, 10.0 * i, 10.0 * i) for i in range(6)]  # frames 0..5
    up = [BallSample(6 + i, 50.0 + 10.0 * i, 50.0 - 10.0 * (i + 1)) for i in range(6)]
    return down + up


def test_straight_trajectory_has_no_events() -> None:
    assert detect_direction_changes(_straight()) == []


def test_sharp_turn_is_detected() -> None:
    events = detect_direction_changes(_v_shape(), min_turn_deg=45)
    assert len(events) >= 1
    idx = [i for i, _ in events]
    assert any(4 <= i <= 6 for i in idx)  # turn near the apex


def test_slow_jitter_below_speed_is_ignored() -> None:
    # Tiny wobble under the speed threshold -> not an event.
    samples = [BallSample(i, 50.0 + (i % 2), 50.0) for i in range(12)]
    assert detect_direction_changes(samples, min_speed_px=3.0) == []


def test_classify_shot_when_wrist_near() -> None:
    samples = _v_shape()
    events = detect_direction_changes(samples, min_turn_deg=45)
    # A player's wrist is always near -> every event is a shot by player 3.
    out = classify_events(samples, events, wrist_near=lambda _f, _xy: 3)
    assert out and all(e.kind == "shot" and e.player_id == 3 for e in out)


def test_classify_bounce_when_no_player() -> None:
    samples = _v_shape()
    events = detect_direction_changes(samples, min_turn_deg=45)
    out = classify_events(samples, events, wrist_near=lambda _f, _xy: None)
    assert out and all(e.kind == "bounce" and e.player_id is None for e in out)


def test_nearby_events_collapse_to_strongest() -> None:
    samples = _v_shape()
    # Two candidate indices one frame apart both counted -> dedup keeps one.
    events = [(5, 120.0), (6, 90.0)]
    out = classify_events(samples, events, wrist_near=lambda _f, _xy: None, guard=3)
    assert len(out) == 1
    assert out[0].turn_deg == 120.0  # the stronger turn survives

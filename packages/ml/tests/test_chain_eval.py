from padel_ml.chain_eval import Row, same_team, summarize


def _row(
    truth_s: float | None,
    detected_s: float | None,
    truth_type: str | None = None,
    truth_player: int | None = None,
    player: int | None = None,
    shot_type: str | None = None,
) -> Row:
    return Row(
        "r",
        "T",
        truth_s,
        detected_s,
        truth_type,
        truth_player,
        player,
        shot_type,
    )


def test_detection_counts_real_hits_and_false_alarms() -> None:
    rows = [
        _row(1.0, 1.05),  # found
        _row(2.0, None),  # missed
        _row(None, 3.0),  # false alarm
    ]
    numbers = summarize(rows)
    assert numbers["detection_precision"] == 0.5
    assert numbers["detection_recall"] == 0.5


def test_other_is_left_out_of_type_but_not_of_detection() -> None:
    rows = [_row(1.0, 1.0, "Other", shot_type="Smash"), _row(2.0, 2.0, "Smash", shot_type="Smash")]
    numbers = summarize(rows)
    assert numbers["type_accuracy"] == 1.0
    assert numbers["detection_recall"] == 1.0


def test_end_to_end_needs_detection_player_and_type() -> None:
    rows = [
        _row(1.0, 1.0, "Forehand", 1, player=1, shot_type="Forehand"),  # all right
        _row(2.0, 2.0, "Forehand", 3, player=4, shot_type="Forehand"),  # wrong player
        _row(3.0, None, "Smash", 2),  # missed
    ]
    numbers = summarize(rows)
    assert abs(numbers["end_to_end_vigo"] - 1 / 3) < 1e-3
    assert abs(numbers["end_to_end_when_what"] - 2 / 3) < 1e-3
    assert numbers["team_accuracy"] == 1.0  # 4 is on 3's team


def test_teams_follow_the_papers_numbering() -> None:
    assert same_team(1, 2) and same_team(3, 4)
    assert not same_team(2, 3)
    assert not same_team(None, 1)

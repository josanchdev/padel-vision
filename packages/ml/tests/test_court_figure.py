import numpy as np
import pytest
from padel_ml.thesis.court_figure import (
    Player,
    darken_outside,
    metres,
    net_distance,
    partner_distances,
)


def _player(number: int, x: float, y: float) -> Player:
    return Player(number, np.zeros((17, 3), np.float32), (0.0, 0.0, 1.0, 1.0), (x, y))


def test_partners_are_measured_within_each_team() -> None:
    players = [
        _player(1, 2.0, 15.0),
        _player(2, 8.0, 15.0),
        _player(3, 2.0, 3.0),
        _player(4, 5.0, 7.0),
    ]
    assert partner_distances(players) == {(1, 2): 6.0, (3, 4): 5.0}


def test_a_team_with_a_missing_player_has_no_partner_distance() -> None:
    assert partner_distances([_player(1, 2.0, 15.0), _player(3, 2.0, 3.0)]) == {}


@pytest.mark.parametrize(("y", "expected"), [(14.5, 4.5), (3.0, 7.0), (10.0, 0.0)])
def test_net_distance_is_straight_across_on_either_side(y: float, expected: float) -> None:
    assert net_distance(_player(1, 3.0, y)) == pytest.approx(expected)


def test_distances_are_written_with_one_decimal_and_a_decimal_comma() -> None:
    assert metres(4.4349) == r"4,4\,m"
    assert metres(6.96) == r"7,0\,m"


def test_only_the_outside_of_the_mask_is_darkened() -> None:
    image = np.full((100, 100, 3), 200, np.uint8)
    square = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])
    masked = darken_outside(image, square)
    assert (masked[50, 50] == 200).all()
    assert (masked[5, 5] < 100).all()
    assert (image == 200).all(), "the input frame must be left untouched"

from pathlib import Path

import numpy as np
from padel_ml.thesis.confusion_figure import cell_text, comma, draw_confusion_figure, row_shares


def test_decimals_use_a_comma() -> None:
    assert comma(0.8035, 3) == "0,804"


def test_each_row_adds_up_to_one() -> None:
    shares = row_shares([[642, 123, 34, 0], [1, 4, 3, 87]])
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert np.isclose(shares[0, 0], 642 / 799)


def test_a_cell_prints_its_share_over_its_hits() -> None:
    assert cell_text(642 / 799, 642) == "0,804\n(642)"


def test_the_figure_is_written(tmp_path: Path) -> None:
    out = tmp_path / "matriz.pdf"
    draw_confusion_figure([[3, 1], [0, 4]], ["A", "B"], out)
    assert out.stat().st_size > 0

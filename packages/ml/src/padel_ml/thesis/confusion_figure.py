"""The confusion matrix figure of section 5.1.4: where the shot classifier goes wrong.

The usual confusion matrix, drawn the way everyone recognises it (as scikit-learn
or MLflow draw it) and not in the style of the other figures: rows are the true
class, columns the predicted one, the colour is the share of the row, and each
cell prints that share with three decimals over the number of hits. Decimals use
a comma, as the rest of the dissertation does; the title goes in the caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from padel_ml.thesis.latex import fixed

CMAP = "Blues"
TEXT_SWITCH = 0.5
"""Share above which a cell is dark enough to need white text."""


def comma(value: float, places: int) -> str:
    """`value` with `places` decimals and a decimal comma: 0.8035 -> 0,804.

    Rounded half-up like the tables (`padel_ml.thesis.latex.fixed`), so a cell
    and the table print the same digits for the same value."""
    return fixed(value, places).replace(".", ",")


def row_shares(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Each count over its row total: the recall of the class on the diagonal."""
    counts = np.asarray(matrix, dtype=np.float64)
    shares: npt.NDArray[np.float64] = counts / counts.sum(axis=1, keepdims=True)
    return shares


def cell_text(share: float, hits: int) -> str:
    """What one cell prints: the share of its row, and the hits below it."""
    return f"{comma(share, 3)}\n({hits})"


def draw_confusion_figure(matrix: list[list[int]], labels: list[str], out: Path) -> None:
    """Write the matrix of `matrix` (rows true, columns predicted) to `out`."""
    counts = np.asarray(matrix, dtype=np.int64)
    shares = row_shares(counts)
    with mpl.rc_context({"font.size": 11}):
        figure = Figure(figsize=(6.4, 5.2))
        axes = figure.subplots()
        image = axes.imshow(shares, cmap=CMAP, vmin=0.0, vmax=1.0)
        colorbar = figure.colorbar(image, ax=axes)
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 1)))
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                share = float(shares[row, column])
                axes.text(
                    column,
                    row,
                    cell_text(share, int(counts[row, column])),
                    ha="center",
                    va="center",
                    color="white" if share > TEXT_SWITCH else "black",
                )
        positions = range(len(labels))
        axes.set_xticks(positions, labels)
        axes.set_yticks(positions, labels)
        axes.set_xlabel("Predicho")
        axes.set_ylabel("Real")
        figure.tight_layout()
        figure.savefig(out)

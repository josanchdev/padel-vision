"""The data figures of the dissertation, as vector PDF set by LaTeX itself.

Drawn at their final width (the text block is 412.28 pt) and rendered through
matplotlib's PGF backend with the memoria's own preamble, so the labels are in
Latin Modern with the decimal comma, like the text around them, and the PDF is
included without rescaling (memoria/GUIA.md, section 5).

Colours (checked with the dataviz palette validator on a white page):
- the four shot types use four validated categorical slots, discards in grey;
- magnitudes (confusion matrices) use a single blue ramp;
- a reference that is not ours (Decorte et al.) is drawn in neutral grey.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.typing import RcKeyType

from padel_ml.thesis.latex import fixed
from padel_ml.thesis.sources import Evidence

TEXT_WIDTH_IN = 412.27864 / 72.27
"""\\textwidth of the memoria, measured on the compiled template."""

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
REFERENCE = "#9a9893"
"""Neutral grey for a result that is not ours."""

SERIES = ["#2a78d6", "#eb6834", "#4a3aa7"]
"""Categorical slots for up to three series (validated all-pairs on white)."""

SHOT_COLOURS = {
    "Forehand": "#2a78d6",
    "Backhand": "#eb6834",
    "Smash": "#4a3aa7",
    "Serve": "#1baf7a",
    "Other": "#c3c2b7",
}
"""Validated adjacent order for stacked bars; the serve's low contrast on white
is why these figures always carry a legend with the names."""

SHOT_NAMES = {
    "Forehand": "Derecha",
    "Backhand": "Revés",
    "Smash": "Remate",
    "Serve": "Saque",
    "Other": "Descartes",
}

TOURNAMENT_NAMES = {
    "VIGO": "Vigo",
    "BRUSSEL": "Bruselas",
    "VALLADOLID": "Valladolid",
    "VALENCIA": "Valencia",
    "MALAGA": "Málaga",
    "FINLAND": "Finlandia",
    "MADRID": "Madrid",
    "DUITSLAND": "Alemania",
    "AMSTERDAM": "Ámsterdam",
    "MENORCA": "Menorca",
    "MALMO": "Malmö",
}

MATRIX_PLACES = 2
"""Confusion matrix cells: two decimals (0,80), the usual choice and what
scikit-learn prints by default; three would crowd sixteen cells."""

BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PREAMBLE = "\n".join(
    [
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{siunitx}",
        r"\sisetup{output-decimal-marker={,}}",
    ]
)

STYLE: dict[RcKeyType, Any] = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": PREAMBLE,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.6,
    "axes.labelcolor": SECONDARY_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "text.color": INK,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
}


def decimal_axis(places: int) -> FuncFormatter:
    """Tick labels with a decimal comma: 0.95 -> 0,95."""
    return FuncFormatter(lambda value, _: fixed(value, places).replace(".", "{,}"))


def _save(figure: Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, backend="pgf")


def _ink_for(fill: str | tuple[float, float, float, float]) -> str:
    """White text on dark cells, ink on light ones."""
    red, green, blue = to_rgb(fill)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.5 else INK


def _tournament(key: str) -> str:
    return TOURNAMENT_NAMES[key.split("_", 1)[1]]


def threshold_figure(evidence: Evidence, out: Path) -> None:
    """F1, precision and recall of the audio detector by threshold, 3 runs."""
    data = evidence("audio_threshold_seeds")
    by_threshold = data["params"]["by_threshold"]
    chosen = float(data["params"]["threshold"])
    # At 0.9 the detector finds nothing (F1 0): left out so the useful range is legible.
    thresholds = sorted(float(t) for t, row in by_threshold.items() if row["f1_mean"] > 0)
    rows = [by_threshold[f"{t:.1f}"] for t in thresholds]
    figure = Figure(figsize=(TEXT_WIDTH_IN, 2.4), layout="constrained")
    axes = figure.add_subplot()
    series = [
        ("F1", "f1_mean", SERIES[0]),
        ("\\emph{Precision}", "precision_mean", SERIES[1]),
        ("\\emph{Recall}", "recall_mean", SERIES[2]),
    ]
    for label, key, colour in series:
        values = [row[key] for row in rows]
        axes.plot(thresholds, values, color=colour, marker="o", markersize=3.5, label=label)
        axes.annotate(
            label,
            (thresholds[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
        )
    spread = np.array([row["f1_std"] for row in rows])
    f1 = np.array([row["f1_mean"] for row in rows])
    axes.fill_between(thresholds, f1 - spread, f1 + spread, color=SERIES[0], alpha=0.15, lw=0)
    axes.axvline(chosen, color=MUTED_INK, linewidth=0.8)
    axes.annotate(
        "umbral elegido",
        (chosen, 0.78),
        xytext=(4, 3),
        textcoords="offset points",
        va="bottom",
        color=SECONDARY_INK,
    )
    axes.set_xlabel("Umbral de detección")
    axes.set_ylabel("Valor (media de 3 ejecuciones)")
    axes.set_xlim(thresholds[0] - 0.02, thresholds[-1] + 0.12)
    axes.set_ylim(0.78, 1.0)
    axes.set_xticks(thresholds)
    axes.xaxis.set_major_formatter(decimal_axis(1))
    axes.yaxis.set_major_formatter(decimal_axis(2))
    axes.grid(axis="y")
    axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3)
    _save(figure, out)


def _confusion(
    axes: Axes, matrix: np.ndarray[Any, np.dtype[np.int64]], rows: list[str], cols: list[str]
) -> None:
    """Each cell is the share of its row: of the real Derecha, which part is
    predicted as each class (the diagonal is the recall). Each row label carries
    how many hits that share is taken over."""
    totals = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    colour_map = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    axes.imshow(share, cmap=colour_map, vmin=0, vmax=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            fill = colour_map(share[i, j])
            text = fixed(share[i, j], MATRIX_PLACES).replace(".", "{,}")
            axes.text(j, i, text, ha="center", va="center", color=_ink_for(fill))
    axes.set_xticks(range(len(cols)), cols)
    labels = [f"{row} ({int(n)})" for row, n in zip(rows, totals[:, 0], strict=True)]
    axes.set_yticks(range(len(rows)), labels)
    axes.set_xlabel("Predicción")
    axes.set_ylabel("Real (golpes)")
    axes.xaxis.set_label_position("top")
    axes.xaxis.tick_top()
    for label in axes.get_xticklabels():
        label.set_verticalalignment("baseline")  # «Saque» and «Remate» on one line
    for spine in axes.spines.values():
        spine.set_visible(False)


def shot_type_confusion_figure(evidence: Evidence, out: Path) -> None:
    """What kind of shot: real type against the classifier's, by tournament."""
    data = evidence("shot_type_classifier")
    matrix = np.array(data["confusion"], dtype=np.int64)
    figure = Figure(figsize=(0.6 * TEXT_WIDTH_IN, 2.6), layout="constrained")
    _confusion(figure.add_subplot(), matrix, data["labels"], data["labels"])
    _save(figure, out)


def dataset_figure(evidence: Evidence, out: Path) -> None:
    """Labelled hits of each tournament, stacked by shot type."""
    per_tournament = evidence("dataset_per_tournament")
    order = sorted(per_tournament)  # tournament keys start with their date
    positions = np.arange(len(order))[::-1]
    figure = Figure(figsize=(TEXT_WIDTH_IN, 3.2), layout="constrained")
    axes = figure.add_subplot()
    left = np.zeros(len(order))
    for shot, colour in SHOT_COLOURS.items():
        counts = np.array([per_tournament[key].get(shot, 0) for key in order], dtype=float)
        axes.barh(positions, counts, left=left, height=0.6, color=colour, edgecolor="white",
                  linewidth=0.8, label=SHOT_NAMES[shot])  # fmt: skip
        left += counts
    axes.set_yticks(positions, [_tournament(k) for k in order])
    axes.set_xlabel("Golpes etiquetados")
    axes.grid(axis="x")
    axes.spines["left"].set_visible(False)
    axes.legend(loc="lower right", ncols=5, bbox_to_anchor=(1.0, 1.0))
    _save(figure, out)


FIGURES: dict[str, Callable[[Evidence, Path], None]] = {
    "umbral-detector.pdf": threshold_figure,
    "tipo-confusion.pdf": shot_type_confusion_figure,
    "dataset-por-torneo.pdf": dataset_figure,
}
"""Output name -> function that draws it."""


def draw_all(evidence: Evidence, directory: Path) -> list[Path]:
    """Every figure into `directory`, set with the memoria's fonts."""
    written = []
    with mpl.rc_context(STYLE):
        for name, draw in FIGURES.items():
            draw(evidence, directory / name)
            written.append(directory / name)
    return written

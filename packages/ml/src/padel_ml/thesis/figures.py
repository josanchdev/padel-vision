"""Shared style of the dissertation's figures: colours, the matplotlib style and the
text width.

The figures are drawn at their final width and rendered through matplotlib's PGF
backend with the memoria's own preamble, so the labels are in Latin Modern with the
decimal comma, like the text around them, and the PDF is included without rescaling
(memoria/GUIA.md, section 5). Each figure lives in its own module; what they share
is here.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.typing import RcKeyType

from padel_cv.arrays import ImageArray

TEXT_WIDTH_IN = 455.24411 / 72.27
"""\\textwidth of the memoria: A4 with 2.5 cm margins (160 mm)."""

CORNER_IN = 0.05
PIXELS_WIDE = 1200
"""Largest raster width embedded per panel: about 500 dpi at its printed size.
Smaller images keep their own pixels."""

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

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


def _aspect(image: ImageArray) -> float:
    return float(image.shape[1]) / float(image.shape[0])


def image_panel(
    figure: Figure, image: ImageArray, left_in: float, width_in: float, bottom_in: float = 0.0
) -> Axes:
    """An RGB image at (`left_in`, `bottom_in`) inches, with rounded corners.

    The axes keep the image's own pixel coordinates, whatever raster is embedded."""
    height_in = width_in / _aspect(image)
    width, height = figure.get_size_inches()
    axes = figure.add_axes(
        (left_in / width, bottom_in / height, width_in / width, height_in / height)
    )
    shown = image
    if image.shape[1] > PIXELS_WIDE:
        rows = round(PIXELS_WIDE / _aspect(image))
        shown = cast(
            ImageArray, cv2.resize(image, (PIXELS_WIDE, rows), interpolation=cv2.INTER_AREA)
        )
    picture = axes.imshow(
        shown, extent=(0, image.shape[1], image.shape[0], 0), interpolation="none"
    )
    outline = FancyBboxPatch(
        (left_in, bottom_in),
        width_in,
        height_in,
        boxstyle=f"round,pad=0,rounding_size={CORNER_IN}",
        transform=figure.dpi_scale_trans,
        facecolor="none",
        edgecolor=BASELINE,
        linewidth=0.4,
    )
    picture.set_clip_path(outline)
    figure.add_artist(outline)
    axes.set_axis_off()
    return axes

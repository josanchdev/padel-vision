"""The problem figure of chapter 2: a point, and the same point processed.

Two frames of the same instant side by side: as the video arrives, and as the
system delivers it, drawn by its own renderer (skeletons, ball trail, the hit's
player and type, the running tally). Nothing is drawn for the figure itself.

Set at the memoria's text width through the PGF backend, so the two headings
are typeset by LaTeX in Latin Modern, like the page around them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from padel_cv.arrays import ImageArray
from padel_ml.thesis.figures import BASELINE, SECONDARY_INK, STYLE

TEXT_WIDTH_IN = 455.24411 / 72.27
"""\\textwidth of the memoria: A4 with 2.5 cm margins (160 mm)."""

GAP_IN = 0.14
HEADER_IN = 0.24
CORNER_IN = 0.05
PIXELS_WIDE = 1200
"""Each panel's raster width: about 390 dpi at its printed size."""

HEADINGS = (r"\textbf{Entrada}: un punto", r"\textbf{Salida}: el punto procesado")


def _panel(figure: Figure, image: ImageArray, left_in: float, width_in: float) -> None:
    """An RGB image at `left_in` (inches), bottom-aligned, with rounded corners."""
    height_in = width_in * image.shape[0] / image.shape[1]
    width, height = figure.get_size_inches()
    axes = figure.add_axes((left_in / width, 0.0, width_in / width, height_in / height))
    rows = round(PIXELS_WIDE * image.shape[0] / image.shape[1])
    shown = cv2.resize(image, (PIXELS_WIDE, rows), interpolation=cv2.INTER_AREA)
    picture = axes.imshow(shown, interpolation="none")
    outline = FancyBboxPatch(
        (left_in, 0.0),
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


def draw_problem_figure(raw: ImageArray, processed: ImageArray, out: Path) -> None:
    """Write the figure to `out` (PDF). Both frames are RGB and of the same instant."""
    if raw.shape != processed.shape:
        raise ValueError(f"frames differ in size: {raw.shape} and {processed.shape}")
    panel_in = (TEXT_WIDTH_IN - GAP_IN) / 2
    image_in = panel_in * raw.shape[0] / raw.shape[1]
    height_in = image_in + HEADER_IN

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, height_in))
        for left_in, image, heading in zip(
            (0.0, panel_in + GAP_IN), (raw, processed), HEADINGS, strict=True
        ):
            _panel(figure, image, left_in, panel_in)
            figure.text(
                left_in / TEXT_WIDTH_IN,
                (image_in + 0.07) / height_in,
                heading,
                fontsize=8.5,
                color=SECONDARY_INK,
                ha="left",
                va="bottom",
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

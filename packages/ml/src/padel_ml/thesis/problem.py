"""The problem figure of chapter 2: a point, the same point processed, and a detail.

Two frames of the same instant side by side: as the video arrives, and as the
system delivers it, drawn by its own renderer (skeletons, ball trail, the hit's
player and type, the running tally). At the printed size the renderer's labels
are too small to read, so a third panel shows the hit's region of the processed
frame at its own resolution, outlined where it comes from. Nothing is drawn on
the frames for the figure except that outline.

Set at the memoria's text width through the PGF backend, so the headings are
typeset by LaTeX in Latin Modern, like the page around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from padel_cv.arrays import ImageArray
from padel_ml.thesis.figures import BASELINE, SECONDARY_INK, STYLE

TEXT_WIDTH_IN = 455.24411 / 72.27
"""\\textwidth of the memoria: A4 with 2.5 cm margins (160 mm)."""

GAP_IN = 0.14
HEADER_IN = 0.24
CORNER_IN = 0.05
PIXELS_WIDE = 1200
"""Largest raster width embedded per panel: about 500 dpi at its printed size.
Smaller images, like the detail, keep their own pixels."""

HEADINGS = (
    r"\textbf{Entrada}: punto original",
    r"\textbf{Salida}: punto procesado",
    r"\textbf{Detalle}",
)


@dataclass(frozen=True)
class Region:
    """A rectangle of the frame, in its pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    def of(self, image: ImageArray) -> ImageArray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


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


def draw_problem_figure(raw: ImageArray, processed: ImageArray, detail: Region, out: Path) -> None:
    """Write the figure to `out` (PDF). Both frames are RGB and of the same instant."""
    if raw.shape != processed.shape:
        raise ValueError(f"frames differ in size: {raw.shape} and {processed.shape}")
    zoom = detail.of(processed)
    # The three panels share one height and fill the text width between them.
    image_in = (TEXT_WIDTH_IN - 2 * GAP_IN) / (2 * _aspect(raw) + _aspect(zoom))
    frame_in = image_in * _aspect(raw)
    lefts = (0.0, frame_in + GAP_IN, 2 * (frame_in + GAP_IN))
    height_in = image_in + HEADER_IN

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, height_in))
        panels = [
            image_panel(figure, image, left_in, image_in * _aspect(image))
            for image, left_in in zip((raw, processed, zoom), lefts, strict=True)
        ]
        panels[1].add_patch(
            Rectangle(
                (detail.x0, detail.y0),
                detail.x1 - detail.x0,
                detail.y1 - detail.y0,
                facecolor="none",
                edgecolor="white",
                linewidth=0.8,
            )
        )
        for left_in, heading in zip(lefts, HEADINGS, strict=True):
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

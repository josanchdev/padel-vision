"""The ball heatmap figure of section 4.2.3: what the ball network returns.

What goes in and what comes out, for one instant:

(a) The frame around the ball, untouched: the ball as the network sees it, a
    few pixels next to the player.
(b) The heatmap of that same part of the frame, on its own. It covers the whole
    image, one value per cell, and almost every cell is 0: it lights up in one
    place only.
(c) The cells around the ball, enlarged, each with its value written in: the
    ball is a patch of cells, not a single point, and the system keeps the cell
    with the highest value (outlined), provided it reaches 0.5.

The three sit in a row, one height. The same white square marks the enlarged
cells in (a) and (b) without covering the ball.

The heatmap is computed here the way the system's `BallDetector` computes it
(three consecutive frames at 768x432, through the trained network), without
going through the detector, which only returns the final position.

Colours follow the spectrogram figure (`magma`, black to yellow), so the two
colour figures of the chapter read alike.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from padel_cv.arrays import ImageArray
from padel_ml.thesis.figures import INK, STYLE, TEXT_WIDTH_IN

FloatArray = npt.NDArray[np.float32]

INPUT_SIZE = (768, 432)
"""Frame size the system feeds the network (`BallDetector`'s default)."""
THRESHOLD = 0.5
"""Lowest peak the system accepts as a ball (`BallDetector`'s default)."""
COLORMAP = "magma"

WINDOW_CELLS = 7
"""Cells per side of the enlarged view: the ball's patch and where it fades."""
CROP_PX = (560, 400)
"""Part of the frame shown in (a) and (b), centred on the ball: enough to keep
the nearest player in view, at the video's own pixels (no resampling)."""

GAP_IN = 0.16
CAPTION_IN = 0.3


def ball_heatmap(
    model: Callable[[torch.Tensor], torch.Tensor], frames: list[ImageArray]
) -> FloatArray:
    """The heatmap the network returns for the last of three consecutive BGR frames."""
    stacked = np.concatenate(
        [
            np.transpose(
                cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0,
                (2, 0, 1),
            )
            for frame in frames
        ],
        axis=0,
    )
    with torch.no_grad():
        output = model(torch.from_numpy(stacked)[None])
    return cast(FloatArray, output[0, 0].cpu().numpy().astype(np.float32))


def peak_cell(heatmap: FloatArray) -> tuple[int, int]:
    """(column, row) of the highest cell: where the system puts the ball."""
    row, column = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return int(column), int(row)


def window_origin(
    heatmap: FloatArray, centre: tuple[int, int], size: int = WINDOW_CELLS
) -> tuple[int, int]:
    """Top-left cell of a `size` x `size` window around `centre`, kept inside the map."""
    rows, columns = heatmap.shape
    column = min(max(centre[0] - size // 2, 0), columns - size)
    row = min(max(centre[1] - size // 2, 0), rows - size)
    return column, row


def crop_origin(frame_shape: tuple[int, ...], centre_px: tuple[float, float]) -> tuple[int, int]:
    """Top-left pixel of the `CROP_PX` part of the frame centred on the ball."""
    height, width = frame_shape[:2]
    x = min(max(round(centre_px[0] - CROP_PX[0] / 2), 0), width - CROP_PX[0])
    y = min(max(round(centre_px[1] - CROP_PX[1] / 2), 0), height - CROP_PX[1])
    return x, y


def decimal(value: float) -> str:
    """0.97 -> 0,97."""
    return f"{value:.2f}".replace(".", ",")


def _cells(axes: Axes, window: FloatArray) -> None:
    """Panel (c): the enlarged cells with their values and the peak outlined."""
    size = window.shape[0]
    axes.imshow(window, cmap=COLORMAP, vmin=0.0, vmax=1.0, interpolation="none")
    for row in range(size):
        for column in range(size):
            value = float(window[row, column])
            axes.text(
                column,
                row,
                decimal(value),
                ha="center",
                va="center",
                fontsize=6.0,
                color=INK if value >= 0.55 else "white",
            )
    peak_column, peak_row = peak_cell(window)
    axes.add_patch(
        Rectangle(
            (peak_column - 0.5, peak_row - 0.5),
            1,
            1,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.3,
        )
    )
    axes.set_xlim(-0.5, size - 0.5)
    axes.set_ylim(size - 0.5, -0.5)
    axes.set_xticks([])
    axes.set_yticks([])
    for edge in axes.spines.values():
        edge.set_linewidth(0.5)


def draw_heatmap_figure(frame_bgr: ImageArray, heatmap: FloatArray, out: Path) -> None:
    """Write the figure to `out` (PDF): the frame, its heatmap and the cells enlarged."""
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    cell_px = frame.shape[1] / heatmap.shape[1]
    peak = peak_cell(heatmap)
    peak_px = ((peak[0] + 0.5) * cell_px, (peak[1] + 0.5) * cell_px)
    x0, y0 = crop_origin(frame.shape, peak_px)
    crop = frame[y0 : y0 + CROP_PX[1], x0 : x0 + CROP_PX[0]]
    column, row = window_origin(heatmap, peak)
    window = heatmap[row : row + WINDOW_CELLS, column : column + WINDOW_CELLS]
    # The heatmap cells under the crop, placed in the crop's own pixels.
    first_column, first_row = int(x0 // cell_px), int(y0 // cell_px)
    last_column = int(np.ceil((x0 + CROP_PX[0]) / cell_px))
    last_row = int(np.ceil((y0 + CROP_PX[1]) / cell_px))
    under = heatmap[first_row:last_row, first_column:last_column]
    under_extent = (
        first_column * cell_px - x0,
        last_column * cell_px - x0,
        last_row * cell_px - y0,
        first_row * cell_px - y0,
    )
    zoom = (column * cell_px - x0, row * cell_px - y0, WINDOW_CELLS * cell_px)

    # Three panels in a row, one height, filling the text width.
    aspect = CROP_PX[0] / CROP_PX[1]
    height_in = (TEXT_WIDTH_IN - 2 * GAP_IN) / (2 * aspect + 1.0)
    frame_in = height_in * aspect
    total_in = height_in + CAPTION_IN

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, total_in))

        def box(left_in: float, width_in: float) -> tuple[float, float, float, float]:
            return (
                left_in / TEXT_WIDTH_IN,
                CAPTION_IN / total_in,
                width_in / TEXT_WIDTH_IN,
                height_in / total_in,
            )

        lefts = (0.0, frame_in + GAP_IN, 2 * (frame_in + GAP_IN))
        photo = figure.add_axes(box(lefts[0], frame_in))
        photo.imshow(crop, interpolation="none")
        heat = figure.add_axes(box(lefts[1], frame_in))
        heat.imshow(
            under,
            cmap=COLORMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="none",
            extent=under_extent,
        )
        for axes in (photo, heat):
            axes.set_xlim(0, CROP_PX[0])
            axes.set_ylim(CROP_PX[1], 0)
            axes.set_axis_off()
            axes.add_patch(
                Rectangle(
                    zoom[:2], zoom[2], zoom[2], facecolor="none", edgecolor="white", linewidth=0.7
                )
            )
        cells = figure.add_axes(box(lefts[2], height_in))
        _cells(cells, window)

        for left_in, width_in, caption in zip(
            lefts, (frame_in, frame_in, height_in), ("(a)", "(b)", "(c)"), strict=True
        ):
            figure.text(
                (left_in + width_in / 2) / TEXT_WIDTH_IN,
                0.02 / total_in,
                caption,
                fontsize=8.5,
                color=INK,
                ha="center",
                va="bottom",
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

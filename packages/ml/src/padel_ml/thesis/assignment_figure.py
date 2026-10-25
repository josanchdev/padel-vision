"""The hit assignment figure of section 4.2.4: why distances are measured in heights.

One frame of a smash by a near-side player, at the instant of the hit, beside a
table. The frame shows, for each of the four players, the line from the ball to
the point the vote measures to (the closer wrist, or the box centre when no wrist
is seen) and the player's height in the image. The table gives, per player, that
distance in pixels, the height, and the distance in heights, which is what the
vote compares (`padel_ml.hit_assignment.player_distance`).

The example is picked so that the two disagree: in pixels the ball is nearest to
a far-side player, who looks small; in heights it is nearest to the hitter.

The frame is a PDF set through the PGF backend (raster image, vector marks); the
table is LaTeX, so it shares the memoria's fonts and table style. A small .tex
file puts the two side by side, for the figure environment to \\input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from padel_cv import overlay
from padel_cv.arrays import ImageArray
from padel_cv.detections import PoseDetection
from padel_ml.hit_assignment import player_distance
from padel_ml.thesis.figures import STYLE, TEXT_WIDTH_IN
from padel_ml.thesis.latex import fixed, tabular

LEFT_WRIST, RIGHT_WRIST = 9, 10
MIN_WRIST_CONFIDENCE = 0.3
"""The wrist confidence the vote requires (`padel_ml.hit_assignment`)."""

FRAME_SHARE = 0.52
"""Share of the text width the frame takes; the table gets the rest."""

LINE = "black"
"""The measures drawn on the frame: thin black lines, for every player alike, so
they inform without pulling the eye away from the players."""
LINE_WIDTH = 0.7
LETTER = "#e8141e"
"""Red for the letters d and h, the only marks meant to be read."""


@dataclass(frozen=True)
class Measure:
    """What the vote reads for one player in one frame."""

    player: int
    point: tuple[float, float]
    """Where the distance is measured to: the closer wrist, or the box centre."""
    pixels: float
    height: float
    """Height of the player's box in the image."""
    heights: float
    """`pixels / height`: the distance the vote compares."""


def measured_point(pose: PoseDetection, ball: tuple[float, float]) -> tuple[float, float]:
    """The point of the player the vote measures to, as `player_distance` chooses it."""
    wrists = [
        (float(pose.keypoints[w, 0]), float(pose.keypoints[w, 1]))
        for w in (LEFT_WRIST, RIGHT_WRIST)
        if pose.keypoints[w, 2] >= MIN_WRIST_CONFIDENCE
    ]
    if wrists:
        return min(wrists, key=lambda w: float(np.hypot(w[0] - ball[0], w[1] - ball[1])))
    x1, y1, x2, y2 = pose.bbox_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def measure(poses: list[PoseDetection], ball: tuple[float, float]) -> list[Measure]:
    """Pixels, height and heights for every identified player, J1 first."""
    out = []
    for pose in sorted(poses, key=lambda p: p.player_id or 0):
        heights = player_distance(pose, ball)
        if pose.player_id is None or heights is None:
            continue
        height = max(pose.bbox_xyxy[3] - pose.bbox_xyxy[1], 1.0)
        point = measured_point(pose, ball)
        pixels = float(np.hypot(point[0] - ball[0], point[1] - ball[1]))
        out.append(Measure(pose.player_id, point, pixels, height, heights))
    return out


def decimal(value: float, places: int) -> str:
    """Plain text with a decimal comma, so that \\textbf can make it bold."""
    return fixed(value, places).replace(".", ",")


def table(measures: list[Measure]) -> str:
    """Per player: d (px), h (px) and d/h, the smallest of each comparison in bold."""
    nearest_px = min(measures, key=lambda m: m.pixels).player
    nearest_h = min(measures, key=lambda m: m.heights).player

    def bold(text: str, on: bool) -> str:
        return rf"\textbf{{{text}}}" if on else text

    rows: list[list[str] | str] = [
        [
            rf"J{m.player}",
            bold(decimal(m.pixels, 0), m.player == nearest_px),
            decimal(m.height, 0),
            bold(decimal(m.heights, 2), m.player == nearest_h),
        ]
        for m in measures
    ]
    return tabular("lccc", ["", "$d$ (px)", "$h$ (px)", "$d/h$"], rows)


def _letter(axes: Axes, x: float, y: float, text: str) -> None:
    """A small label for a measure, on a white patch so it reads on any background."""
    axes.text(
        x,
        y,
        text,
        fontsize=7.5,
        color=LETTER,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none"},
    )


def draw_frame(
    frame_bgr: ImageArray,
    poses: list[PoseDetection],
    ball: tuple[float, float],
    crop: tuple[int, int, int, int],
    out: Path,
    labelled: int,
) -> None:
    """The cropped frame with skeletons, the measured lines and each player's height.

    Only player `labelled` (the hitter) carries the letters d and h: once is enough
    to name the two measures, and the other players stay uncluttered."""
    x0, y0, x1, y1 = crop
    image = frame_bgr.copy()
    for pose in poses:
        if pose.player_id is not None:
            overlay.skeleton(
                image,
                pose.keypoints,
                overlay.COCO_SKELETON,
                overlay.PLAYER_COLOURS[pose.player_id],
                thickness=3,
                joint_radius=4,
            )
    rgb = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    width_in = TEXT_WIDTH_IN * FRAME_SHARE
    height_in = width_in * (y1 - y0) / (x1 - x0)
    by_id = {p.player_id: p for p in poses}

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(width_in, height_in))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.imshow(rgb, extent=(x0, x1, y1, y0), interpolation="none")
        for m in measure(poses, ball):
            axes.plot(
                [ball[0], m.point[0]],
                [ball[1], m.point[1]],
                color=LINE,
                linewidth=LINE_WIDTH,
                linestyle=(0, (3, 1.5)),
            )
            if m.player == labelled:
                _letter(axes, (ball[0] + m.point[0]) / 2 + 16, (ball[1] + m.point[1]) / 2, "$d$")
            bx1, by1, bx2, by2 = by_id[m.player].bbox_xyxy
            side = bx2 + 8
            axes.plot([side, side], [by1, by2], color=LINE, linewidth=LINE_WIDTH)
            for y in (by1, by2):
                axes.plot([side - 5, side + 5], [y, y], color=LINE, linewidth=LINE_WIDTH)
            if m.player == labelled:
                _letter(axes, side + 16, by1 + 0.75 * (by2 - by1), "$h$")
            axes.text(
                bx1,
                by1 - 8,
                rf"J{m.player}",
                fontsize=6.5,
                color=LINE,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.18,rounding_size=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            )
        axes.scatter(
            [ball[0]],
            [ball[1]],
            s=42,
            facecolors="none",
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )
        axes.set_xlim(x0, x1)
        axes.set_ylim(y1, y0)
        axes.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")


def figure_tex(frame_pdf: str, measures: list[Measure]) -> str:
    """The frame and the table side by side, for `\\input` inside a figure environment."""
    return "\n".join(
        [
            "% Generado por scripts/build_assignment_figure.py: no editar a mano.",
            rf"\begin{{minipage}}[c]{{{FRAME_SHARE:.2f}\linewidth}}",
            rf"  \includegraphics[width=\linewidth]{{{frame_pdf}}}",
            r"\end{minipage}\hfill",
            rf"\begin{{minipage}}[c]{{{1 - FRAME_SHARE - 0.03:.2f}\linewidth}}",
            r"  \centering",
            "  " + table(measures).replace("\n", "\n  "),
            r"\end{minipage}",
            "",
        ]
    )

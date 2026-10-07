"""The problem figure of chapter 2: one point, what goes in and what comes out.

The left frame is the video exactly as the system receives it. The right one is
the same frame with what the system returns at that instant: who is who, and the
hit's player and type. Below runs the start of the point's timeline, one mark
per detected hit, with the hit of the frame joined to its player. The hits are
the full system's output with models trained without the point's tournament
(docs/metrics/chain_cv_hits.csv), so the figure shows what it does on a
tournament it has not seen.

Drawn at the memoria's text width through the PGF backend, so its text is set
by LaTeX in Latin Modern, like the page around it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Ellipse, FancyBboxPatch

from padel_cv import overlay
from padel_cv.arrays import ImageArray
from padel_cv.detections import PoseDetection
from padel_ml.thesis.figures import BASELINE, INK, MUTED_INK, SECONDARY_INK, STYLE

TEXT_WIDTH_IN = 455.24411 / 72.27
"""\\textwidth of the memoria: A4 with 2.5 cm margins (160 mm)."""

TYPE_NAMES = {"Forehand": "Derecha", "Backhand": "Revés", "Smash": "Remate", "Serve": "Saque"}
MARKERS = {"circle": "o", "diamond": "D", "triangle": "^", "square": "s"}
MARKER_SIZES = {"o": 6.0, "D": 5.0, "^": 6.5, "s": 5.0}
"""Point sizes that make the four shapes look equally heavy."""

GAP_IN = 0.14
HEADER_IN = 0.24
TIMELINE_IN = 0.56
LEGEND_IN = 0.24
CORNER_IN = 0.05
FADE_S = 0.9
"""The timeline's last stretch fades out: the point goes on past the figure."""
LEAD_S = 0.35
"""Room before the start of the point, so the first tick label is not cut."""
PIXELS_WIDE = 1100
"""Each panel's raster width: about 360 dpi at its printed size."""


@dataclass(frozen=True)
class Hit:
    time_s: float
    player: int
    shot_type: str


@dataclass(frozen=True)
class Crop:
    """The part of the 1920x1080 frame that is shown: the court and its players."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def aspect(self) -> float:
        return (self.x1 - self.x0) / (self.y1 - self.y0)

    def box(self, player: PoseDetection) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = player.bbox_xyxy
        return x0 - self.x0, y0 - self.y0, x1 - self.x0, y1 - self.y0


def system_hits(rally: str, hits_csv: Path) -> list[Hit]:
    """The hits the full system returned for `rally`, in time order."""
    with hits_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter=";") if row["rally"] == rally]
    hits = [
        Hit(float(row["detected_s"]), int(row["player"]), row["shot_type"])
        for row in rows
        if row["detected_s"] and row["player"] and row["shot_type"]
    ]
    return sorted(hits, key=lambda hit: hit.time_s)


def spotlight(image: ImageArray, box: tuple[float, float, float, float]) -> ImageArray:
    """Dim the frame around `box`, the hitter, with a soft edge."""
    x0, y0, x1, y1 = box
    centre = (round((x0 + x1) / 2), round((y0 + y1) / 2))
    axes = (round((x1 - x0) * 1.2), round((y1 - y0) * 0.8))
    mask = np.zeros(image.shape[:2], np.float32)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    soft = np.asarray(cv2.GaussianBlur(mask, (0, 0), sigmaX=(x1 - x0) * 0.6), np.float32)
    factor = 1.0 - 0.45 * (1.0 - soft)
    return (image.astype(np.float32) * factor[..., None]).astype(np.uint8)


def _fade(time_s: float, end_s: float) -> float:
    """Opacity along the timeline: full, then down to zero over the last FADE_S."""
    return float(np.clip((end_s - time_s) / FADE_S, 0.0, 1.0))


def _panel(
    figure: Figure, image: ImageArray, left_in: float, bottom_in: float, width_in: float
) -> Axes:
    """An image axes at a place given in inches, with rounded corners."""
    height_in = width_in * image.shape[0] / image.shape[1]
    width, height = figure.get_size_inches()
    axes = figure.add_axes(
        (left_in / width, bottom_in / height, width_in / width, height_in / height)
    )
    shown = cv2.resize(
        image,
        (PIXELS_WIDE, round(PIXELS_WIDE / image.shape[1] * image.shape[0])),
        interpolation=cv2.INTER_AREA,
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


def _tag(axes: Axes, x: float, y: float, text: str, colour: str, size: float) -> None:
    axes.text(
        x,
        y,
        text,
        fontsize=size,
        color="white",
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.5",
            "facecolor": colour,
            "edgecolor": "none",
        },
    )


def draw_problem_figure(
    frame: ImageArray,
    players: list[PoseDetection],
    hits: list[Hit],
    shown: Hit,
    crop: Crop,
    out: Path,
) -> None:
    """Write the figure to `out` (PDF). `frame` is RGB; `players` are those of its instant."""
    hitter = next(player for player in players if player.player_id == shown.player)
    raw = np.ascontiguousarray(frame[crop.y0 : crop.y1, crop.x0 : crop.x1])
    lit = spotlight(raw, crop.box(hitter))

    panel_in = (TEXT_WIDTH_IN - GAP_IN) / 2
    image_in = panel_in / crop.aspect
    height_in = LEGEND_IN + TIMELINE_IN + image_in + HEADER_IN
    images_bottom = LEGEND_IN + TIMELINE_IN

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, height_in))
        _panel(figure, raw, 0.0, images_bottom, panel_in)
        outputs = _panel(figure, lit, panel_in + GAP_IN, images_bottom, panel_in)
        for left_in, header in (
            (0.0, r"\textbf{Entrada}: el vídeo del punto, sin anotar"),
            (
                panel_in + GAP_IN,
                r"\textbf{Salida}: instante, jugador y tipo de cada golpe",
            ),
        ):
            figure.text(
                left_in / TEXT_WIDTH_IN,
                (images_bottom + image_in + 0.07) / height_in,
                header,
                fontsize=8.5,
                color=SECONDARY_INK,
                ha="left",
                va="bottom",
            )

        # Who is who on the output frame; the hitter gets a ring and the verdict.
        for player in players:
            if player.player_id is None:
                continue
            x0, y0, x1, y1 = crop.box(player)
            colour = overlay.PLAYER_HEX[player.player_id]
            if player is hitter:
                feet, width = ((x0 + x1) / 2, y1 - 4), (x1 - x0) * 1.3
                outputs.add_patch(
                    Ellipse(
                        feet, width, width * 0.3, facecolor=colour, alpha=0.25, edgecolor="none"
                    )
                )
                outputs.add_patch(
                    Ellipse(
                        feet, width, width * 0.3, facecolor="none", edgecolor=colour, linewidth=1.2
                    )
                )
                verdict = (
                    rf"J{player.player_id}\,\textperiodcentered\,{TYPE_NAMES[shown.shot_type]}"
                )
                _tag(outputs, (x0 + x1) / 2, y0 - 12, rf"\textbf{{{verdict}}}", colour, 8.5)
            else:
                _tag(outputs, (x0 + x1) / 2, y0 - 10, rf"J{player.player_id}", colour, 7.5)

        # The timeline: aligned so the shown hit sits right under its hitter.
        timeline = figure.add_axes((0.0, LEGEND_IN / height_in, 1.0, TIMELINE_IN / height_in))
        timeline.set_axis_off()
        hitter_x_in = (
            panel_in
            + GAP_IN
            + panel_in * ((crop.box(hitter)[0] + crop.box(hitter)[2]) / 2) / (crop.x1 - crop.x0)
        )
        end_s = -LEAD_S + (shown.time_s + LEAD_S) * TEXT_WIDTH_IN / hitter_x_in
        timeline.set_xlim(-LEAD_S, end_s)
        timeline.set_ylim(-1.0, 1.0)

        steps = np.linspace(0.0, end_s, 200)
        segments = np.stack(
            [
                np.column_stack([steps[:-1], np.zeros(199)]),
                np.column_stack([steps[1:], np.zeros(199)]),
            ],
            axis=1,
        )
        colours = [mpl.colors.to_rgba(BASELINE, _fade(t, end_s)) for t in steps[1:]]
        timeline.add_collection(LineCollection(list(segments), colors=colours, linewidths=0.9))
        for second in range(int(end_s) + 1):
            alpha = _fade(second, end_s)
            timeline.plot(
                [second, second], [-0.12, 0.0], color=BASELINE, linewidth=0.6, alpha=alpha
            )
            if abs(second - shown.time_s) > 0.6:  # the shown hit carries its own time there
                timeline.text(
                    second,
                    -0.32,
                    rf"\SI{{{second}}}{{s}}",
                    fontsize=7.5,
                    color=MUTED_INK,
                    alpha=alpha,
                    ha="center",
                    va="top",
                )

        for hit in hits:
            if hit.time_s > end_s:
                continue
            alpha = _fade(hit.time_s, end_s)
            marker = MARKERS[overlay.SHOT_SHAPES[hit.shot_type]]
            colour = overlay.PLAYER_HEX[hit.player]
            focus = hit == shown
            size = MARKER_SIZES[marker] * (1.5 if focus else 1.0)
            if focus:
                timeline.plot(
                    hit.time_s,
                    0.0,
                    marker="o",
                    markersize=size * 2.1,
                    color=colour,
                    alpha=0.18,
                    zorder=2,
                )
            timeline.plot(
                hit.time_s,
                0.0,
                marker=marker,
                markersize=size,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=0.8,
                alpha=alpha,
                zorder=3,
            )
            timeline.text(
                hit.time_s,
                0.42,
                rf"\textbf{{J{hit.player}}}" if focus else rf"J{hit.player}",
                fontsize=7.5,
                color=INK if focus else SECONDARY_INK,
                alpha=alpha,
                ha="center",
                va="bottom",
            )
            if focus:
                timeline.text(
                    hit.time_s,
                    -0.32,
                    rf"\SI{{{hit.time_s:.2f}}}{{s}}",
                    fontsize=7.5,
                    color=INK,
                    ha="center",
                    va="top",
                )

        # The shown hit joined to its hitter on the output frame.
        x0, _, x1, y1 = crop.box(hitter)
        figure.add_artist(
            ConnectionPatch(
                xyA=(shown.time_s, 0.0),
                coordsA=timeline.transData,
                xyB=((x0 + x1) / 2, crop.y1 - crop.y0),
                coordsB=outputs.transData,
                color=overlay.PLAYER_HEX[shown.player],
                linewidth=0.7,
                linestyle=(0, (2.0, 1.5)),
                shrinkA=6.0,
                shrinkB=0.0,
            )
        )

        handles = [Line2D([], [], linestyle="none")]
        labels = [r"\textbf{Jugador}"]
        for number, colour in overlay.PLAYER_HEX.items():
            handles.append(
                Line2D([], [], linestyle="none", marker="o", markersize=5.0, color=colour)
            )
            labels.append(f"J{number}")
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(r"\quad\textbf{Golpe}")
        for shot_type, name in TYPE_NAMES.items():
            marker = MARKERS[overlay.SHOT_SHAPES[shot_type]]
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    markersize=MARKER_SIZES[marker] * 0.85,
                    color=MUTED_INK,
                )
            )
            labels.append(name)
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=8,
            frameon=False,
            handlelength=0.9,
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0.1,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

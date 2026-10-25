"""The player figure of section 4.2.2: the court mask and the positions in metres.

Two views of one instant of a point, side by side:

(a) The frame with what the system keeps of it: the court mask (the padded
    trapezoid, everything outside it darkened) and the four identified players,
    each skeleton in its player's colour, the same colours as the web viewer.
(b) The same instant on the court plane: each player where the homography puts
    the midpoint of their ankles, with the distance between partners and from
    each player to the net. It is the layout the dataset's paper (Decorte et al.)
    uses to show its positional analysis, drawn here from our own output.

Distances carry one decimal: the hand-marked court reprojects with an error of
about a decimetre, so a second decimal would claim a precision the system does
not have.

The frame is a raster at its own resolution; the mask outline, the player tags
and the whole court map are vector, set by LaTeX through the PGF backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from padel_cv import overlay
from padel_cv.arrays import ImageArray
from padel_cv.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from padel_ml.thesis.figures import INK, SECONDARY_INK, STYLE, TEXT_WIDTH_IN, image_panel

OUTSIDE_BRIGHTNESS = 0.3
"""What is left of the image outside the mask: dark enough to read as discarded,
light enough to still show the stands it discards."""

SERVICE_LINE_M = 6.95
"""Distance from the net to each service line (padel regulations)."""

GAP_IN = 0.32
TICKS_IN = 0.22
"""Room on the left of the court map for its tick labels."""
CAPTION_IN = 0.36
"""Room under the panels for the map's tick labels and the (a)/(b) labels."""
MARGIN_M = 0.6
"""Court map padding around the 10 x 20 m court, so dots on a line are whole."""

COURT_FILL = "#eaeef5"
COURT_LINE = "#9ba5b6"
NET_LINE = "#4f5868"
DASHES = (0, (3.0, 2.0))
LABEL_SIZE = 7.0


@dataclass(frozen=True)
class Player:
    """One identified player at the instant shown."""

    number: int
    """1-4, the J of the system."""
    keypoints: npt.NDArray[np.float32]
    """(17, 3) COCO keypoints in frame pixels, with their confidence."""
    bbox_xyxy: tuple[float, float, float, float]
    position_m: tuple[float, float]
    """Ankle midpoint projected by the homography, in court metres."""


def partner_distances(players: list[Player]) -> dict[tuple[int, int], float]:
    """Metres between the two players of each team (J1-J2 far side, J3-J4 near side)."""
    at = {player.number: np.array(player.position_m) for player in players}
    return {
        (a, b): float(np.linalg.norm(at[a] - at[b]))
        for a, b in ((1, 2), (3, 4))
        if a in at and b in at
    }


def net_distance(player: Player) -> float:
    """Metres from the player to the net, straight across the court."""
    return abs(player.position_m[1] - NET_Y_M)


def metres(value: float) -> str:
    """4.43 -> 4,4\\,m: one decimal, decimal comma, thin space before the unit."""
    return f"{value:.1f}".replace(".", ",") + r"\,m"


def darken_outside(image: ImageArray, polygon: npt.NDArray[np.float64]) -> ImageArray:
    """A copy of `image` with everything outside `polygon` darkened."""
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.round(polygon).astype(np.int32)], 1)
    dimmed = (image.astype(np.float32) * OUTSIDE_BRIGHTNESS).astype(np.uint8)
    return np.where(inside[..., None] == 1, image, dimmed).astype(np.uint8)


def masked_frame(
    frame_bgr: ImageArray, polygon: npt.NDArray[np.float64], players: list[Player]
) -> ImageArray:
    """Panel (a) as RGB: the mask applied and the skeletons drawn in their colours.

    Strokes are drawn heavier than in the processed video: the frame is printed a
    few centimetres wide, where the video's 2-pixel limbs would vanish."""
    image = darken_outside(frame_bgr, polygon)
    for player in players:
        overlay.skeleton(
            image,
            player.keypoints,
            overlay.COCO_SKELETON,
            overlay.PLAYER_COLOURS[player.number],
            thickness=4,
            joint_radius=5,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _label(axes: Axes, x: float, y: float, text: str) -> None:
    """A distance on its dashed line, on a patch of court so the dashes stop short of it."""
    axes.text(
        x,
        y,
        text,
        fontsize=LABEL_SIZE,
        color=INK,
        ha="center",
        va="center",
        zorder=5,
        bbox={"boxstyle": "square,pad=0.12", "facecolor": COURT_FILL, "edgecolor": "none"},
    )


def _court_map(axes: Axes, players: list[Player]) -> None:
    """Panel (b): the court from above, far side up, as the camera sees it."""
    axes.add_patch(
        Rectangle(
            (0, 0),
            COURT_WIDTH_M,
            COURT_LENGTH_M,
            facecolor=COURT_FILL,
            edgecolor=COURT_LINE,
            linewidth=0.8,
        )
    )
    for y in (NET_Y_M - SERVICE_LINE_M, NET_Y_M + SERVICE_LINE_M):
        axes.plot([0, COURT_WIDTH_M], [y, y], color=COURT_LINE, linewidth=0.6)
    axes.plot(
        [COURT_WIDTH_M / 2] * 2,
        [NET_Y_M - SERVICE_LINE_M, NET_Y_M + SERVICE_LINE_M],
        color=COURT_LINE,
        linewidth=0.6,
    )
    axes.plot([-0.35, COURT_WIDTH_M + 0.35], [NET_Y_M] * 2, color=NET_LINE, linewidth=1.3)

    at = {player.number: player.position_m for player in players}
    for (a, b), distance in partner_distances(players).items():
        (xa, ya), (xb, yb) = at[a], at[b]
        axes.plot([xa, xb], [ya, yb], color=SECONDARY_INK, linewidth=0.7, linestyle=DASHES)
        _label(axes, (xa + xb) / 2, (ya + yb) / 2, metres(distance))
    for player in players:
        x, y = player.position_m
        axes.plot([x, x], [y, NET_Y_M], color=SECONDARY_INK, linewidth=0.7, linestyle=DASHES)
        _label(axes, x, (y + NET_Y_M) / 2, metres(net_distance(player)))
    for player in players:
        x, y = player.position_m
        axes.scatter(
            [x],
            [y],
            s=62,
            color=overlay.PLAYER_HEX[player.number],
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
        axes.text(
            x,
            y,
            rf"\textbf{{{player.number}}}",
            fontsize=5.5,
            color="white",
            ha="center",
            va="center",
            zorder=7,
        )

    axes.set_xlim(-MARGIN_M, COURT_WIDTH_M + MARGIN_M)
    axes.set_ylim(-MARGIN_M, COURT_LENGTH_M + MARGIN_M)
    axes.set_aspect("equal")
    axes.set_xticks([0, 5, 10])
    axes.set_yticks([0, 5, 10, 15, 20])
    axes.tick_params(labelsize=LABEL_SIZE, pad=2)
    for edge in axes.spines.values():
        edge.set_visible(False)


def draw_court_figure(
    frame_bgr: ImageArray,
    polygon: npt.NDArray[np.float64],
    players: list[Player],
    out: Path,
) -> None:
    """Write the figure to `out` (PDF): the masked frame and the court map, one height."""
    frame = masked_frame(frame_bgr, polygon, players)
    aspect = frame.shape[1] / frame.shape[0]
    map_aspect = (COURT_WIDTH_M + 2 * MARGIN_M) / (COURT_LENGTH_M + 2 * MARGIN_M)
    # Both panels share one height and fill the text width between them.
    height_in = (TEXT_WIDTH_IN - GAP_IN - TICKS_IN) / (aspect + map_aspect)
    frame_in = height_in * aspect
    map_left = frame_in + GAP_IN + TICKS_IN
    map_in = height_in * map_aspect
    total_in = height_in + CAPTION_IN

    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(TEXT_WIDTH_IN, total_in))
        photo = image_panel(figure, frame, 0.0, frame_in, bottom_in=CAPTION_IN)
        photo.add_patch(
            Polygon(polygon, closed=True, facecolor="none", edgecolor="white", linewidth=0.7)
        )
        for player in players:
            x1, y1, _, _ = player.bbox_xyxy
            photo.text(
                x1,
                y1 - 14,
                rf"J{player.number}",
                fontsize=6.0,
                color="white",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.18,rounding_size=0.25",
                    "facecolor": overlay.PLAYER_HEX[player.number],
                    "edgecolor": "none",
                },
            )
        court = figure.add_axes(
            (
                map_left / TEXT_WIDTH_IN,
                CAPTION_IN / total_in,
                map_in / TEXT_WIDTH_IN,
                height_in / total_in,
            )
        )
        _court_map(court, players)
        for centre_in, caption in (
            (frame_in / 2, "(a)"),
            (map_left + map_in / 2, "(b)"),
        ):
            figure.text(
                centre_in / TEXT_WIDTH_IN,
                0.02 / total_in,
                caption,
                fontsize=8.5,
                color=INK,
                ha="center",
                va="bottom",
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

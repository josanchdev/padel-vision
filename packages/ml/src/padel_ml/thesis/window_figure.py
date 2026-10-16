"""The adaptive window figure of section 4.2.5: what stretch of a point the classifier sees.

A schematic in the manner of BST's figure 1 (Chang, CVPRW 2026), panels (c) and
(d) in one: a time bar from the previous hit to the next one, plus the margin
epsilon in another colour; the player whose hit is classified, in black, before,
at and after the hit; the rivals, in red, at the two hits that bound the window;
and the ball as dashed arcs from racket to racket, each in the colour of whoever
hit it. The players are pictograms with a padel racket, not real poses: the
figure is about the window, and the text explains the rest.

Only epsilon is to scale (the system's 0,25 s against the spacing between hits);
the spacing itself is illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, Rectangle

from padel_ml.shot_type_dataset import DEFAULT_T_S, EPSILON_RATIO
from padel_ml.thesis.figures import INK, STYLE

Point = npt.NDArray[np.float64]

RIVAL = "#e8141e"
MARGIN = "#1baf7a"
BALL = "#e2e23c"

EPSILON_S = EPSILON_RATIO * DEFAULT_T_S
"""The tail past the next hit, as the system cuts it."""
BEFORE_S, AFTER_S = 1.0, 1.3
"""Illustrative time from the previous hit to this one, and from this one to the next."""

WIDTH_IN = 3.9
TORSO_IN = 0.12
"""One torso length on the page, in inches; the poses below are in torso lengths."""
LIMB_PT = 1.9
BAR_IN = 0.09
BALL_IN = 0.034
FONT_PT = 8

HEAD_RADIUS, NECK = 0.27, 0.45
HANDLE, FACE_LENGTH, FACE_WIDTH = 0.25, 0.8, 0.62


@dataclass(frozen=True)
class Pose:
    """A side-view pictogram: hips at the origin, y up, in torso lengths.

    Each arm is (elbow, hand) from the shoulder and each leg (knee, foot) from
    the hips; the racket is in `racket_arm` and continues its forearm."""

    shoulder: tuple[float, float]
    arms: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    legs: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    racket_arm: int = 0

    def mirrored(self) -> Pose:
        def flip(p: tuple[float, float]) -> tuple[float, float]:
            return (-p[0], p[1])

        return Pose(
            flip(self.shoulder),
            tuple((flip(e), flip(h)) for e, h in self.arms),
            tuple((flip(k), flip(f)) for k, f in self.legs),
            self.racket_arm,
        )


RUN = Pose(
    shoulder=(0.25, 0.97),
    arms=(((0.6, 0.62), (0.95, 0.85)), ((-0.12, 0.55), (-0.4, 0.3))),
    legs=(((0.5, -0.45), (0.42, -1.2)), ((-0.18, -0.6), (-0.7, -0.95))),
)
"""Moving, racket ready in front."""
SMASH = Pose(
    shoulder=(-0.05, 1.0),
    arms=(((0.12, 1.55), (0.25, 2.1)), ((0.45, 1.25), (0.7, 1.6))),
    legs=(((0.22, -0.62), (0.18, -1.3)), ((-0.28, -0.6), (-0.42, -1.27))),
)
"""Hitting overhead, the free arm pointing at the ball."""
FOREHAND = Pose(
    shoulder=(0.12, 0.98),
    arms=(((0.6, 0.85), (1.08, 0.78)), ((-0.35, 0.72), (-0.72, 0.85))),
    legs=(((0.58, -0.5), (0.65, -1.22)), ((-0.42, -0.55), (-0.9, -1.05))),
)
"""Reaching for a ball at shoulder height."""
VOLLEY = Pose(
    shoulder=(0.08, 0.98),
    arms=(((0.5, 1.25), (0.8, 1.65)), ((-0.35, 0.65), (-0.6, 0.45))),
    legs=(((0.35, -0.6), (0.4, -1.28)), ((-0.25, -0.6), (-0.45, -1.25))),
)
"""Meeting the ball above the head, in front."""


def _xy(p: tuple[float, float]) -> Point:
    return np.array(p, dtype=np.float64)


def lowest(pose: Pose) -> float:
    """How far below the hips the feet reach, in torso lengths."""
    return -min(foot[1] for _, foot in pose.legs)


def racket_tip(pose: Pose) -> tuple[Point, Point]:
    """Centre of the racket face and the direction it points, in torso lengths."""
    elbow, hand = (_xy(p) for p in pose.arms[pose.racket_arm])
    along = (hand - elbow) / float(np.hypot(*(hand - elbow)))
    return hand + along * (HANDLE + FACE_LENGTH / 2), along


def draw_player(axes: Axes, pose: Pose, feet: Point, colour: str) -> Point:
    """Draw `pose` standing at `feet` (inches); returns where the ball meets its racket."""

    def page(p: tuple[float, float] | Point) -> Point:
        lifted = np.asarray(p, dtype=np.float64) + np.array([0.0, lowest(pose)])
        return np.asarray(feet + lifted * TORSO_IN, dtype=np.float64)

    line: dict[str, Any] = {"color": colour, "solid_capstyle": "round", "zorder": 3}
    hips, shoulder = page((0.0, 0.0)), page(pose.shoulder)
    axes.plot(*zip(hips, shoulder, strict=True), linewidth=LIMB_PT * 1.7, **line)
    for elbow, hand in pose.arms:
        joints = np.array([shoulder, page(elbow), page(hand)])
        axes.plot(joints[:, 0], joints[:, 1], linewidth=LIMB_PT, **line)
    for knee, foot in pose.legs:
        joints = np.array([hips, page(knee), page(foot)])
        axes.plot(joints[:, 0], joints[:, 1], linewidth=LIMB_PT, **line)

    axis = _xy(pose.shoulder) / float(np.hypot(*pose.shoulder))
    head = page(_xy(pose.shoulder) + axis * NECK)
    axes.add_patch(
        Circle(
            tuple(head),
            HEAD_RADIUS * TORSO_IN,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=3.5,
        )
    )

    centre, along = racket_tip(pose)
    grip = _xy(pose.arms[pose.racket_arm][1])
    axes.plot(
        *zip(page(grip), page(grip + along * HANDLE), strict=True),
        linewidth=LIMB_PT * 0.8,
        **line,
    )
    angle = float(np.degrees(np.arctan2(along[1], along[0])))
    axes.add_patch(
        Ellipse(
            tuple(page(centre)),
            FACE_LENGTH * TORSO_IN,
            FACE_WIDTH * TORSO_IN,
            angle=angle,
            color=colour,
            zorder=3,
        )
    )
    # The holes that make it a padel racket and not a strung one.
    across = np.array([-along[1], along[0]])
    for up, side in ((0.0, 0.0), (-0.17, -0.13), (-0.17, 0.13), (0.17, -0.13), (0.17, 0.13)):
        hole = page(centre + along * up + across * side)
        axes.add_patch(Circle(tuple(hole), 0.06 * TORSO_IN, color="white", zorder=4))
    return page(centre + along * FACE_LENGTH / 2) + along * BALL_IN


def arc(axes: Axes, start: Point, end: Point, rise_in: float, colour: str) -> None:
    """The ball's path, dashed and bulging upwards, with the ball halfway along."""
    control = (start + end) / 2 + np.array([0.0, rise_in])
    t = np.linspace(0, 1, 100)[:, None]
    path = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end
    axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=0.8, linestyle=(0, (3, 2)))
    middle = path[len(path) // 2]
    axes.add_patch(
        Circle(tuple(middle), BALL_IN, facecolor=BALL, edgecolor=INK, linewidth=0.5, zorder=5)
    )


def draw_window_figure(out: Path, width_in: float = WIDTH_IN) -> None:
    """Write the figure to `out` (PDF)."""
    previous, hit = 0.0, BEFORE_S
    following, end = hit + AFTER_S, hit + AFTER_S + EPSILON_S
    left_in, right_in = 0.42, 0.25  # room for the end labels, centred on their ticks
    per_s = (width_in - left_in - right_in) / end

    def x(t: float) -> float:
        return left_in + t * per_s

    labels_in, arcs_in = 0.5, 0.22
    tallest = max(lowest(p) + racket_tip(p)[0][1] + FACE_LENGTH / 2 for p in (SMASH, VOLLEY))
    top = BAR_IN / 2 + tallest * TORSO_IN + arcs_in
    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(width_in, top + labels_in))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_xlim(0, width_in)
        axes.set_ylim(-labels_in, top)
        axes.set_aspect("equal")
        axes.set_axis_off()

        bar: dict[str, Any] = {"height": BAR_IN, "edgecolor": "none", "zorder": 1}
        axes.add_patch(
            Rectangle((x(previous), -BAR_IN / 2), x(following) - x(previous), facecolor=INK, **bar)
        )
        axes.add_patch(
            Rectangle((x(following), -BAR_IN / 2), x(end) - x(following), facecolor=MARGIN, **bar)
        )
        for t in (previous, hit, following):
            axes.plot([x(t), x(t)], [-BAR_IN / 2 - 0.05, BAR_IN / 2], color=INK, linewidth=0.7)
        below = -BAR_IN / 2 - 0.08
        for t, text in (
            (previous, "golpe anterior"),
            (hit, "golpe"),
            (following, "golpe siguiente"),
        ):
            axes.text(x(t), below, text, ha="center", va="top", fontsize=FONT_PT)
        axes.text(
            (x(following) + x(end)) / 2,
            below - 0.17,
            rf"$\underbrace{{\hspace{{{x(end) - x(following):.3f}in}}}}_{{\varepsilon}}$",
            ha="center",
            va="top",
            fontsize=FONT_PT,
        )

        feet = BAR_IN / 2
        served = draw_player(axes, FOREHAND, _xy((x(previous), feet)), RIVAL)
        draw_player(axes, RUN, _xy((x(previous + BEFORE_S * 0.5), feet)), INK)
        smashed = draw_player(axes, SMASH, _xy((x(hit), feet)), INK)
        draw_player(axes, RUN, _xy((x(hit + AFTER_S * 0.5), feet)), INK)
        returned = draw_player(axes, VOLLEY.mirrored(), _xy((x(following), feet)), RIVAL)

        arc(axes, served, smashed, arcs_in, RIVAL)
        arc(axes, smashed, returned, arcs_in * 0.6, INK)
        arc(axes, returned, _xy((x(end), feet + 0.12)), 0.08, RIVAL)

        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

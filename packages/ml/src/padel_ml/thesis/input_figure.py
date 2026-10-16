"""The classifier's input figure of section 4.2.5: a hit as the model receives it.

On black, the hitter's skeleton, hip-centred and scaled by the torso, and the ball
in that player's frame of reference. The origin, the hips, is the centre of the
panel. Nothing else of the frame reaches the model: not the court, not the other
players, not the racket.

It is not a reconstruction: it draws the arrays `build_window` returns, the very
function the system runs at inference. The steps of the window around the hit are
drawn fading away from it, a trail that shows the input is a sequence and not a
single pose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from padel_cv.arrays import FloatArray
from padel_ml.shot_type_dataset import ShotWindow
from padel_ml.thesis.figures import STYLE

SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]
"""The COCO skeleton: which of the 17 keypoints are joined."""
LIMB_COLOURS = [mpl.colormaps["hsv"](i / len(SKELETON)) for i in range(len(SKELETON))]
"""One hue per limb, as pose estimators usually draw them."""
BALL = "#e2e23c"

PANEL_IN = 2.6
LIMB_PT = 2.2
MARGIN = 0.15
"""Room around what is drawn, as a fraction of the panel's half-width."""


def step_frames(start: int, end: int, length: int) -> npt.NDArray[np.int64]:
    """The frame behind each step of the window, as `build_window` picks them."""
    return np.round(np.linspace(start, end, length)).astype(np.int64)


def trail(frames: npt.NDArray[np.int64], hit: int, reach: int) -> list[int]:
    """Steps showing the frames within `reach` of the hit, one step per frame."""
    seen: dict[int, int] = {}
    for step, frame in enumerate(frames.tolist()):
        if abs(frame - hit) <= reach and frame not in seen:
            seen[frame] = step
    return [seen[f] for f in sorted(seen)]


def _skeleton(axes: Axes, pose: FloatArray, alpha: float, width: float) -> None:
    """One step's pose, y down as in the image, every keypoint the model receives."""
    for (a, b), colour in zip(SKELETON, LIMB_COLOURS, strict=True):
        axes.plot(
            pose[[a, b], 0],
            pose[[a, b], 1],
            color=colour,
            alpha=alpha,
            linewidth=width,
            solid_capstyle="round",
        )
    axes.scatter(pose[:, 0], pose[:, 1], s=(width * 1.6) ** 2, color="white", alpha=alpha, lw=0)


def _model_view(axes: Axes, window: ShotWindow, steps: list[int], hit_step: int) -> None:
    """The hit step solid, the rest fading with their distance to it.

    The scale is set by the hit step alone, so it does not change with the trail;
    trail steps reaching past the panel are cut at its edge."""
    axes.set_facecolor("black")
    shown = [window.pose[hit_step, :, :2]]
    if window.ball[hit_step, 2] > 0:
        shown.append(window.ball[hit_step, None, :2])
    half = float(np.max(np.abs(np.concatenate(shown)))) * (1 + MARGIN)
    axes.set_xlim(-half, half)
    axes.set_ylim(half, -half)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    furthest = max((abs(s - hit_step) for s in steps), default=0) or 1
    for step in sorted(steps, key=lambda s: -abs(s - hit_step)):
        nearness = 1 - abs(step - hit_step) / (furthest + 1)
        alpha = 1.0 if step == hit_step else 0.12 + 0.4 * nearness
        width = LIMB_PT if step == hit_step else LIMB_PT * 0.7
        _skeleton(axes, window.pose[step], alpha, width)
        if window.ball[step, 2] > 0:
            axes.scatter(
                [window.ball[step, 0]],
                [window.ball[step, 1]],
                s=30 if step == hit_step else 16,
                color=BALL,
                alpha=alpha,
                lw=0,
                zorder=4,
            )


def draw_input_figure(
    window: ShotWindow, steps: list[int], hit_step: int, out: Path, size_in: float = PANEL_IN
) -> None:
    """Write the figure to `out` (PDF); `steps` include `hit_step`."""
    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(size_in, size_in))
        _model_view(figure.add_axes((0, 0, 1, 1)), window, steps, hit_step)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, backend="pgf")

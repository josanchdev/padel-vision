"""Drawing primitives for the analysis overlay.

Split out of the demo script because how the result *reads* is part of the
result: a viewer has to see at a glance who hit and what they hit, without
hunting around the frame.

Three rules, taken from how broadcast graphics do it:

- **Labels live next to the action.** A tag in the corner forces the eye across
  the whole frame; anchored above the player, it is read in place.
- **Text sits on a solid plate**, not on bare outline. Padel footage is blue
  court and white lines, so an outlined glyph competes with the background.
- **Skeletons are thin.** Thick strokes turn a distant player into a blob and
  hide the very gesture being classified.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]

FONT = cv2.FONT_HERSHEY_DUPLEX
"""Duplex over SIMPLEX: same metrics, noticeably cleaner strokes."""

INK = (245, 245, 245)
MUTED = (170, 170, 170)
PLATE = (24, 24, 28)


def text_size(string: str, scale: float, weight: int = 1) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(string, FONT, scale, weight)
    return width, height


def plate(
    image: ImageArray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int] = PLATE,
    alpha: float = 0.82,
    accent: tuple[int, int, int] | None = None,
) -> None:
    """Semi-transparent panel, optionally with a colour bar down its left edge.

    The bar is how the label carries the class colour without tinting the text,
    which would hurt legibility.
    """
    x0, y0 = max(top_left[0], 0), max(top_left[1], 0)
    x1 = min(bottom_right[0], image.shape[1])
    y1 = min(bottom_right[1], image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = (region * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)
    if accent is not None:
        image[y0:y1, x0 : min(x0 + 5, x1)] = accent


def label(
    image: ImageArray,
    string: str,
    anchor: tuple[int, int],
    scale: float = 0.62,
    colour: tuple[int, int, int] = INK,
    accent: tuple[int, int, int] | None = None,
    weight: int = 1,
    centred: bool = False,
) -> tuple[int, int]:
    """Text on a plate. `anchor` is the bottom-left corner (or centre-bottom).

    Returns the plate's top-left, so callers can stack things above it.
    """
    width, height = text_size(string, scale, weight)
    pad_x, pad_y = 11, 8
    x = anchor[0] - width // 2 if centred else anchor[0]
    x = int(np.clip(x, 4, image.shape[1] - width - pad_x * 2 - 4))
    y = int(np.clip(anchor[1], height + pad_y * 2 + 4, image.shape[0] - 4))
    top_left = (x - pad_x, y - height - pad_y * 2)
    bottom_right = (x + width + pad_x, y)
    plate(image, top_left, bottom_right, accent=accent)
    cv2.putText(
        image,
        string,
        (x + (5 if accent else 0), y - pad_y),
        FONT,
        scale,
        colour,
        weight,
        cv2.LINE_AA,
    )
    return top_left


def skeleton(
    image: ImageArray,
    keypoints: npt.NDArray[np.float32],
    edges: list[tuple[int, int]],
    colour: tuple[int, int, int],
    min_confidence: float = 0.3,
    thickness: int = 2,
    joint_radius: int = 3,
) -> None:
    """Thin limbs with small white joints, the way broadcast pose graphics read.

    White joints on a coloured skeleton keep the articulation visible even when
    the limb colour is close to the background.
    """
    for a, b in edges:
        if keypoints[a, 2] >= min_confidence and keypoints[b, 2] >= min_confidence:
            cv2.line(
                image,
                (int(keypoints[a, 0]), int(keypoints[a, 1])),
                (int(keypoints[b, 0]), int(keypoints[b, 1])),
                colour,
                thickness,
                cv2.LINE_AA,
            )
    for joint in range(len(keypoints)):
        if keypoints[joint, 2] >= min_confidence:
            centre = (int(keypoints[joint, 0]), int(keypoints[joint, 1]))
            cv2.circle(image, centre, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)


def ball_trail(
    image: ImageArray,
    positions: list[tuple[float, float] | None],
    colour: tuple[int, int, int] = (60, 240, 255),
) -> None:
    """Fading trail through recent ball positions, newest last.

    A single circle says where the ball is; the trail says where it is *going*,
    which is what makes a smash legible as a smash on a still frame.
    """
    points = [p for p in positions if p is not None]
    if not points:
        return
    for i in range(1, len(points)):
        fade = i / len(points)
        faded = tuple(int(c * (0.45 + 0.55 * fade)) for c in colour)
        cv2.line(
            image,
            (int(points[i - 1][0]), int(points[i - 1][1])),
            (int(points[i][0]), int(points[i][1])),
            faded,
            max(int(2 + 3 * fade), 2),
            cv2.LINE_AA,
        )
    head = points[-1]
    cv2.circle(image, (int(head[0]), int(head[1])), 7, colour, 2, cv2.LINE_AA)


def panel(
    image: ImageArray,
    title: str,
    rows: list[tuple[str, str, tuple[int, int, int]]],
    origin: tuple[int, int] = (28, 28),
    width: int = 260,
) -> None:
    """A titled block of coloured rows — the running tally, broadcast style."""
    line_height = 30
    height = 44 + line_height * len(rows) + 10
    x, y = origin
    plate(image, (x, y), (x + width, y + height), alpha=0.86)
    cv2.putText(image, title, (x + 14, y + 28), FONT, 0.52, MUTED, 1, cv2.LINE_AA)
    cv2.line(image, (x + 14, y + 38), (x + width - 14, y + 38), (70, 70, 78), 1, cv2.LINE_AA)
    for i, (name, value, colour) in enumerate(rows):
        row_y = y + 44 + line_height * i + 20
        cv2.rectangle(image, (x + 14, row_y - 11), (x + 26, row_y + 1), colour, -1)
        cv2.putText(image, name, (x + 36, row_y), FONT, 0.56, INK, 1, cv2.LINE_AA)
        value_width, _ = text_size(value, 0.56, 1)
        cv2.putText(
            image, value, (x + width - 16 - value_width, row_y), FONT, 0.56, INK, 1, cv2.LINE_AA
        )

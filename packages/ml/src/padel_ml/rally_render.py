"""Draw a `RallyAnalysis` over its video: the watchable version of the result.

Rendering is a second pass over the video rather than part of the analysis. The
analysis keeps only detections, so memory stays flat however long the rally.
The previous demo held every decoded frame instead: measured on a 48 s 1080p
rally, 10.6 GB peak against 1.9 GB now, with identical output — and a two-minute
point would not have fitted at all.

Output is H.264 so a browser can play it (the web shows this same file); the
old demo wrote MPEG-4 Part 2, which browsers do not decode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2

from padel_cv import overlay
from padel_cv.arrays import ImageArray
from padel_cv.video_io import H264VideoWriter
from padel_ml.rally_analysis import RallyAnalysis, Shot
from padel_ml.shot_type_dataset import CLASSES

SPANISH = {"Forehand": "DERECHA", "Backhand": "REVES", "Smash": "REMATE", "Serve": "SAQUE"}
# Colour means the player, in the video as in the web viewer (overlay.PLAYER_HEX);
# the stroke type is carried by the label text and by shapes in the tally.
PLAYER_COLOURS = overlay.PLAYER_COLOURS
FLASH_FRAMES = 18
TRAIL_FRAMES = 12


def render_rally(
    analysis: RallyAnalysis,
    out: Path,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Write `out`: skeletons, ball trail, and each shot's verdict over its hitter."""
    shots = {shot.frame_index: shot for shot in analysis.shots}
    out.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(analysis.video))
    flash_until, current = -1, None
    counts: dict[str, int] = {}
    with H264VideoWriter(out, analysis.fps) as writer:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = cast(ImageArray, frame)
            if index in shots:
                flash_until, current = index + FLASH_FRAMES, shots[index]
                if current.shot_type:
                    counts[current.shot_type] = counts.get(current.shot_type, 0) + 1
            _draw_frame(image, index, analysis, current if index <= flash_until else None, counts)
            writer.write(image)
            index += 1
            if on_progress is not None and analysis.n_frames and index % 25 == 0:
                on_progress(min(index / analysis.n_frames, 1.0))
    capture.release()


def _draw_frame(
    image: ImageArray,
    index: int,
    analysis: RallyAnalysis,
    active: Shot | None,
    counts: dict[str, int],
) -> None:
    hitter = active.player_id if active is not None else None
    hitter_pose = None
    for pose in analysis.players.get(index, []):
        assert pose.player_id is not None
        if hitter is not None and pose.player_id == hitter:
            hitter_pose = pose
            continue  # drawn last, so it sits on top of the others
        overlay.skeleton(
            image, pose.keypoints, overlay.COCO_SKELETON, PLAYER_COLOURS[pose.player_id]
        )
        overlay.label(
            image,
            f"J{pose.player_id}",
            (int(pose.bbox_xyxy[0]), int(pose.bbox_xyxy[1]) - 6),
            scale=0.5,
            accent=PLAYER_COLOURS[pose.player_id],
        )

    trail = range(max(index - TRAIL_FRAMES + 1, 0), index + 1)
    overlay.ball_trail(image, [analysis.ball_smoothed.get(f) for f in trail])

    if hitter_pose is not None and active is not None:
        assert hitter_pose.player_id is not None
        # The hitter keeps HIS OWN colour, only drawn heavier: recolouring him by
        # stroke type would break the one thing the colour is for — telling the
        # four players apart — exactly when the viewer is looking hardest. The
        # stroke type is carried by the label instead.
        player_colour = PLAYER_COLOURS[hitter_pose.player_id]
        overlay.skeleton(
            image,
            hitter_pose.keypoints,
            overlay.COCO_SKELETON,
            player_colour,
            thickness=3,
            joint_radius=4,
        )
        # The verdict goes right above the player who produced it: a tag in the
        # corner makes the viewer hunt for who it refers to. Clear of the head,
        # since the box top already sits at the crown.
        x1, y1, x2, _ = hitter_pose.bbox_xyxy
        name = SPANISH.get(active.shot_type or "", "SIN CLASIFICAR")
        top = overlay.label(
            image,
            f"{name}  {active.confidence:.0%}",
            (int((x1 + x2) / 2), int(y1) - 34),
            scale=0.72,
            accent=player_colour,
            centred=True,
        )
        overlay.label(
            image,
            f"JUGADOR {hitter_pose.player_id}",
            (int((x1 + x2) / 2), top[1] - 4),
            scale=0.46,
            accent=player_colour,  # same colour as his skeleton
            centred=True,
        )

    rows = [
        (SPANISH[name], str(counts[name]), overlay.SHOT_SHAPES[name])
        for name in CLASSES
        if counts.get(name)
    ]
    if rows:
        overlay.panel(image, f"GOLPES DETECTADOS   {sum(counts.values())}", rows)

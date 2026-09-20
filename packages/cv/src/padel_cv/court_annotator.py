"""Mark the court by hand, once per tournament (ADR-0015 D2).

The learned detector (court v6, ADR-0006) stays the default — measured over
CVSPORTS it reprojects to 0.11-0.15 m on six of the eleven tournaments, better
than the colour-based method the reference paper discarded. But it drifts to
1-5 m on three and fails outright on two, so those need the manual route the
paper itself uses: the camera is fixed within a tournament, so a handful of
clicks gives an exact homography that serves every rally of that tournament.

Six points rather than four: four corners are enough in theory (a homography has
eight degrees of freedom), but the far corners are often hidden behind the glass,
the frame or a player, and an error there propagates across that whole half. The
two ends of the net line are always clearly visible and anchor the middle of the
court, where play actually happens.

Click order (the on-screen prompt repeats it):
    1. near-left   2. near-right   3. far-right   4. far-left
    5. net-left    6. net-right
"Near" is the bottom of the image (closest to the camera); left/right as seen on
screen.

Keys:  z undo   r restart   ENTER save (needs all six)   ESC cancel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from padel_cv.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M

ImageArray = npt.NDArray[np.uint8]

#: Court-plane coordinates (metres) of each clicked point, in click order.
POINTS_M: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 0.0],  # 1 near-left
        [COURT_WIDTH_M, 0.0],  # 2 near-right
        [COURT_WIDTH_M, COURT_LENGTH_M],  # 3 far-right
        [0.0, COURT_LENGTH_M],  # 4 far-left
        [0.0, NET_Y_M],  # 5 net-left
        [COURT_WIDTH_M, NET_Y_M],  # 6 net-right
    ]
)
PROMPTS = [
    "1/6  esquina CERCANA IZQUIERDA (abajo-izq)",
    "2/6  esquina CERCANA DERECHA (abajo-der)",
    "3/6  esquina LEJANA DERECHA (arriba-der)",
    "4/6  esquina LEJANA IZQUIERDA (arriba-izq)",
    "5/6  RED, extremo IZQUIERDO",
    "6/6  RED, extremo DERECHO",
]
#: Outline drawn while clicking: the court quad, then the net line.
OUTLINE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)]
DISPLAY_WIDTH = 1600


def homography_from_points(points_px: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Least-squares px->m homography from the clicked points.

    With more than four correspondences `findHomography` fits them all, so a
    slightly misplaced corner is averaged out instead of skewing the result.
    """
    count = len(points_px)
    matrix, _ = cv2.findHomography(
        points_px.astype(np.float32), POINTS_M[:count].astype(np.float32)
    )
    return cast(npt.NDArray[np.float64], matrix)


def reprojection_error_m(
    points_px: npt.NDArray[np.float64], homography: npt.NDArray[np.float64]
) -> float:
    """Mean distance (metres) between each clicked point and where it lands."""
    errors = []
    for (x, y), expected in zip(points_px, POINTS_M[: len(points_px)], strict=True):
        vector = homography @ np.array([x, y, 1.0])
        got = np.array([vector[0] / vector[2], vector[1] / vector[2]])
        errors.append(float(np.hypot(*(got - expected))))
    return float(np.mean(errors))


def _text(
    canvas: ImageArray,
    string: str,
    org: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    cv2.putText(canvas, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def _draw(base: ImageArray, points: list[tuple[int, int]], scale: float) -> ImageArray:
    canvas = base.copy()
    height = canvas.shape[0]
    canvas[0:56] = (canvas[0:56] * 0.25).astype(np.uint8)
    done = len(points) == len(PROMPTS)
    prompt = "LISTO — ENTER guarda · z deshace · r reinicia" if done else PROMPTS[len(points)]
    _text(canvas, prompt, (24, 38), 0.9, (80, 220, 255))

    for i, (x, y) in enumerate(points):
        cv2.drawMarker(canvas, (x, y), (60, 240, 60), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 13, (60, 240, 60), 2, cv2.LINE_AA)
        _text(canvas, str(i + 1), (x + 16, y - 12), 0.8, (60, 240, 60))
    for a, b in OUTLINE:
        if a < len(points) and b < len(points):
            cv2.line(canvas, points[a], points[b], (60, 240, 60), 2, cv2.LINE_AA)

    if done:
        # Overlay the service lines from the fitted homography: if they land on
        # the painted lines, the six clicks were right. Cheap visual check that
        # catches a mis-click before it silently corrupts a whole tournament.
        homography = homography_from_points(np.array(points, dtype=np.float64) / scale)
        inverse = np.linalg.inv(homography)
        for y_m in (3.05, 16.95):
            ends = []
            for x_m in (0.0, COURT_WIDTH_M):
                vector = inverse @ np.array([x_m, y_m, 1.0])
                ends.append(
                    (int(vector[0] / vector[2] * scale), int(vector[1] / vector[2] * scale))
                )
            cv2.line(canvas, ends[0], ends[1], (255, 200, 0), 2, cv2.LINE_AA)
        error = reprojection_error_m(np.array(points, dtype=np.float64) / scale, homography)
        _text(
            canvas,
            f"error {error:.3f} m — ¿las lineas naranjas caen sobre las de saque?",
            (24, height - 24),
            0.7,
            (0, 200, 255),
        )
    return canvas


def annotate_court(
    video_path: Path, out_json: Path, frame_index: int = 30
) -> npt.NDArray[np.float64] | None:
    """Show one frame, collect the clicked points, save the homography."""
    from padel_cv.shot_type_annotator import ensure_qt_fonts

    ensure_qt_fonts()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} of {video_path}")

    scale = min(DISPLAY_WIDTH / image.shape[1], 1.0)
    shown = cast(
        ImageArray,
        cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        if scale < 1.0
        else image,
    )

    window = f"Marca la pista - {video_path.stem}"
    points: list[tuple[int, int]] = []
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, _draw(shown, points, scale))
    cv2.waitKey(1)

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(PROMPTS):
            points.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, _draw(shown, points, scale))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("z") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10) and len(points) == len(PROMPTS):  # ENTER
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()

    points_px = np.array(points, dtype=np.float64) / scale
    homography = homography_from_points(points_px)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source_video": str(video_path),
                "frame_index": frame_index,
                "points_px": points_px.tolist(),
                "points_m": POINTS_M.tolist(),
                "corners_px": points_px[:4].tolist(),
                "homography": homography.tolist(),
                "reprojection_error_m": round(reprojection_error_m(points_px, homography), 4),
            },
            indent=2,
        )
        + "\n"
    )
    return homography


def load_corners(path: Path) -> npt.NDArray[np.float64]:
    """The four court corners in pixels, for the player mask."""
    return np.array(json.loads(path.read_text())["corners_px"], dtype=np.float64)

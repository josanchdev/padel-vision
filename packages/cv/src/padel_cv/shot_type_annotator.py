"""Label the TYPE of already-located hits (ADR-0016).

Unlike `shot_annotator`, this tool does not look for hits: the instant comes from
a ground-truth file (CVSPORTS `hits.csv`, or our own marks). The loop is just
jump -> look -> press a key, which is the fast part of labelling.

Keys:
    1 Saque   2 Derecha   3 Reves   4 Remate      o  descartar (Other)
    w  marca/quita "de pared"                     z  deshacer
    <-/->  golpe anterior/siguiente               a/d  frame a frame
    SPACE  reproduce el golpe a camara lenta      r  repite
    q  guardar y salir                            ESC  salir sin guardar

Progress is saved continuously, so closing and reopening resumes where it left.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]

#: ADR-0016: four gesture classes; "Other" is the discard bin (not trained on,
#: used to calibrate the rejection threshold).
SHOT_TYPES: dict[int, str] = {
    ord("1"): "Serve",
    ord("2"): "Forehand",
    ord("3"): "Backhand",
    ord("4"): "Smash",
    ord("o"): "Other",
}
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "Serve": (120, 200, 90),
    "Forehand": (235, 170, 60),
    "Backhand": (90, 160, 240),
    "Smash": (90, 90, 235),
    "Other": (140, 140, 140),
}
_UNSET = (70, 70, 70)
_ACCENT = (235, 190, 80)
_INK = (245, 245, 245)
_MUTED = (165, 165, 165)

PLAY_HALF_FRAMES = 12
"""How many frames either side of the hit `SPACE` replays."""


@dataclass
class TypeMark:
    """One hit: when it happened (given) and what it was (labelled here)."""

    frame: int
    shot_type: str | None = None
    from_wall: bool = False


def load_hit_frames(csv_path: Path, video_name: str, fps: float) -> list[int]:
    """Hit frames for one rally from the CVSPORTS hits.csv (times in seconds)."""
    frames: list[int] = []
    for row in csv.DictReader(csv_path.open()):
        if row["filename"] in (video_name, Path(video_name).stem):
            centre = (float(row["start"]) + float(row["end"])) / 2
            frames.append(round(centre * fps))
    return sorted(frames)


def save_marks(marks: list[TypeMark], csv_path: Path) -> None:
    """Write labels; unlabelled hits are kept with an empty type."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "type", "from_wall"])
        for mark in marks:
            writer.writerow([mark.frame, mark.shot_type or "", int(mark.from_wall)])


def load_marks(csv_path: Path) -> dict[int, TypeMark]:
    """Read back previously saved labels, keyed by frame."""
    if not csv_path.exists():
        return {}
    out: dict[int, TypeMark] = {}
    for row in csv.DictReader(csv_path.open(), delimiter=";"):
        frame = int(row["frame"])
        out[frame] = TypeMark(
            frame=frame,
            shot_type=row["type"] or None,
            from_wall=bool(int(row.get("from_wall") or 0)),
        )
    return out


def _band(canvas: ImageArray, y0: int, y1: int, alpha: float = 0.72) -> None:
    """Darken a horizontal strip so text stays readable over any footage."""
    y0, y1 = max(y0, 0), min(y1, canvas.shape[0])
    if y1 <= y0:
        return
    strip = canvas[y0:y1]
    canvas[y0:y1] = (strip * (1 - alpha)).astype(np.uint8)


def _text(
    canvas: ImageArray,
    string: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = _INK,
    weight: int = 1,
) -> None:
    cv2.putText(
        canvas, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 3, cv2.LINE_AA
    )
    cv2.putText(canvas, string, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_progress(canvas: ImageArray, marks: list[TypeMark], index: int) -> None:
    """A dot per hit: filled in its type colour once labelled, hollow if not.

    Gives an at-a-glance sense of how much is left and where you are, which a
    bare "12/35" counter does not.
    """
    height, width = canvas.shape[:2]
    if not marks:
        return
    margin, y = 40, height - 96
    span = width - 2 * margin
    step = min(span / max(len(marks), 1), 26.0)
    start = margin + (span - step * (len(marks) - 1)) / 2 if len(marks) > 1 else width / 2
    for i, mark in enumerate(marks):
        cx = int(start + i * step)
        color = TYPE_COLORS.get(mark.shot_type or "", _UNSET) if mark.shot_type else _UNSET
        radius = 8 if i == index else 5
        cv2.circle(canvas, (cx, y), radius, color, -1, cv2.LINE_AA)
        if mark.from_wall:  # a wall hit gets a ring
            cv2.circle(canvas, (cx, y), radius + 3, (255, 255, 255), 1, cv2.LINE_AA)
        if i == index:  # caret under the current hit
            cv2.drawMarker(
                canvas, (cx, y + 18), _ACCENT, cv2.MARKER_TRIANGLE_UP, 12, 2, cv2.LINE_AA
            )


def _draw_hud(
    canvas: ImageArray,
    marks: list[TypeMark],
    index: int,
    frame_index: int,
    fps: float,
    saved: bool,
) -> None:
    height, width = canvas.shape[:2]
    mark = marks[index]
    labelled = sum(1 for m in marks if m.shot_type)

    # ---- top bar: where we are and what this hit is
    _band(canvas, 0, 92)
    _text(canvas, f"Golpe {index + 1} / {len(marks)}", (32, 40), 0.85, _INK, 2)
    _text(
        canvas,
        f"{labelled} etiquetados · {len(marks) - labelled} pendientes",
        (32, 72),
        0.55,
        _MUTED,
    )

    if mark.shot_type:
        color = TYPE_COLORS.get(mark.shot_type, _INK)
        label = mark.shot_type + (" · de pared" if mark.from_wall else "")
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        x0 = width - size[0] - 56
        cv2.rectangle(canvas, (x0 - 16, 22), (width - 24, 70), color, -1, cv2.LINE_AA)
        _text(canvas, label, (x0, 56), 0.9, (20, 20, 20), 2)
    else:
        _text(canvas, "sin etiquetar", (width - 240, 56), 0.8, _MUTED, 2)

    # progress bar of the whole rally
    done = labelled / max(len(marks), 1)
    cv2.rectangle(canvas, (0, 88), (width, 92), (60, 60, 60), -1)
    cv2.rectangle(canvas, (0, 88), (int(width * done), 92), _ACCENT, -1)

    _draw_progress(canvas, marks, index)

    # ---- bottom bar: the keys, grouped as they are used
    _band(canvas, height - 68, height)
    keys = [
        ("1", "Saque", TYPE_COLORS["Serve"]),
        ("2", "Derecha", TYPE_COLORS["Forehand"]),
        ("3", "Reves", TYPE_COLORS["Backhand"]),
        ("4", "Remate", TYPE_COLORS["Smash"]),
        ("o", "Descartar", TYPE_COLORS["Other"]),
    ]
    x = 32
    for key, name, color in keys:
        cv2.rectangle(canvas, (x, height - 52), (x + 26, height - 26), color, -1, cv2.LINE_AA)
        _text(canvas, key, (x + 8, height - 33), 0.6, (20, 20, 20), 2)
        _text(canvas, name, (x + 34, height - 33), 0.58, _INK)
        x += 42 + len(name) * 13
    _text(
        canvas,
        "w pared   z deshacer   <- -> golpe   SPACE repetir   q guardar y salir",
        (32, height - 10),
        0.5,
        _MUTED,
    )

    # ---- the moment of contact, marked on the frame itself
    time_s = frame_index / fps
    _text(
        canvas,
        f"frame {frame_index}  ·  {int(time_s // 60):d}:{time_s % 60:05.2f}",
        (width - 300, height - 10),
        0.5,
        _MUTED,
    )
    if frame_index == mark.frame:
        _text(canvas, "IMPACTO", (width // 2 - 60, 130), 0.7, _ACCENT, 2)
    if saved:
        _text(canvas, "guardado", (width - 150, 130), 0.6, (140, 220, 140), 2)


def annotate_types(
    video_path: Path,
    hit_frames: list[int],
    out_csv: Path,
    window: str = "Tipo de golpe",
) -> None:
    """Run the labelling loop over pre-located hits."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    existing = load_marks(out_csv)
    marks = [existing.get(f, TypeMark(frame=f)) for f in hit_frames]
    # resume on the first unlabelled hit
    index = next((i for i, m in enumerate(marks) if not m.shot_type), 0)
    history: list[tuple[int, str | None, bool]] = []

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    frame_index = marks[index].frame
    playing = False
    play_frame = 0
    saved_flash = 0

    def read_at(target: int) -> ImageArray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(target, 0))
        ok, image = capture.read()
        return cast(ImageArray, image) if ok else None

    while True:
        if playing:
            frame_index = marks[index].frame - PLAY_HALF_FRAMES + play_frame
            play_frame += 1
            if play_frame > PLAY_HALF_FRAMES * 2:
                playing = False
                frame_index = marks[index].frame
        image = read_at(frame_index)
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas = image.copy()
        _draw_hud(canvas, marks, index, frame_index, fps, saved_flash > 0)
        saved_flash = max(saved_flash - 1, 0)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(60 if playing else 20) & 0xFF
        if key == 255:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        if key in SHOT_TYPES:
            history.append((index, marks[index].shot_type, marks[index].from_wall))
            marks[index].shot_type = SHOT_TYPES[key]
            save_marks(marks, out_csv)
            saved_flash = 8
            if index < len(marks) - 1:  # advance automatically: keeps the rhythm
                index += 1
                frame_index = marks[index].frame
                playing = False
        elif key == ord("w"):
            history.append((index, marks[index].shot_type, marks[index].from_wall))
            marks[index].from_wall = not marks[index].from_wall
            save_marks(marks, out_csv)
            saved_flash = 8
        elif key == ord("z") and history:
            i, shot_type, from_wall = history.pop()
            marks[i].shot_type, marks[i].from_wall = shot_type, from_wall
            index = i
            frame_index = marks[index].frame
            save_marks(marks, out_csv)
        elif key in (81, ord(",")):  # left arrow
            index = max(index - 1, 0)
            frame_index = marks[index].frame
            playing = False
        elif key in (83, ord(".")):  # right arrow
            index = min(index + 1, len(marks) - 1)
            frame_index = marks[index].frame
            playing = False
        elif key == ord("a"):
            frame_index = max(frame_index - 1, 0)
            playing = False
        elif key == ord("d"):
            frame_index += 1
            playing = False
        elif key in (ord(" "), ord("r")):
            playing = True
            play_frame = 0
        elif key == ord("q"):
            save_marks(marks, out_csv)
            break
        elif key == 27:  # ESC
            break

    capture.release()
    cv2.destroyAllWindows()

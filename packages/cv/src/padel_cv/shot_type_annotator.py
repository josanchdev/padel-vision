"""Label the TYPE of already-located hits (ADR-0016).

Unlike `shot_annotator`, this tool does not look for hits: the instant comes from
a ground-truth file (CVSPORTS `hits.csv`, or our own marks). The loop is just
watch -> press a key, which is the fast part of labelling.

Each hit plays as a LOOPING CLIP, not a still frame. A stroke type is a movement,
and asking a human to name it from one image is both slow and error-prone — the
labels come out noisy and the model inherits that noise. The model itself sees a
whole window (ADR-0016 F), so the annotator should too.

Keys:
    1 Saque   2 Derecha   3 Reves   4 Remate      o  descartar (Other)
    z  deshacer
    <-/->  golpe anterior/siguiente               a/d  frame a frame
    SPACE  pausa/reanuda el bucle                 r  reinicia el clip
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

DISPLAY_WIDTH = 1280
"""Frames are scaled to this width for display (1080p does not fit on screen)."""

CLIP_HALF_FRAMES = 25
"""Frames either side of the hit shown as a looping clip (~1 s each way at 25
fps): enough to see wind-up, contact and follow-through, which is what tells a
forehand from a backhand — one still frame does not."""


@dataclass
class TypeMark:
    """One hit: when it happened (given) and what it was (labelled here)."""

    frame: int
    shot_type: str | None = None


def load_hit_frames(csv_path: Path, video_name: str, fps: float) -> list[int]:
    """Hit frames for one rally from the CVSPORTS hits.csv (times in seconds)."""
    frames: list[int] = []
    for row in csv.DictReader(csv_path.open()):
        if row["filename"] in (video_name, Path(video_name).stem):
            centre = (float(row["start"]) + float(row["end"])) / 2
            frames.append(round(centre * fps))
    return sorted(frames)


def save_marks(marks: list[TypeMark], csv_path: Path) -> None:
    """Write labels; unlabelled hits are kept with an empty type.

    `from_wall` stays as a column (always 0) so the format matches the older
    shot CSVs, but it is no longer annotated: a wall rebound is meant to be
    derived from the ball trajectory, not typed in by hand (see ADR-0016 C).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "type", "from_wall"])
        for mark in marks:
            writer.writerow([mark.frame, mark.shot_type or "", 0])


def load_marks(csv_path: Path) -> dict[int, TypeMark]:
    """Read back previously saved labels, keyed by frame."""
    if not csv_path.exists():
        return {}
    out: dict[int, TypeMark] = {}
    for row in csv.DictReader(csv_path.open(), delimiter=";"):
        frame = int(row["frame"])
        out[frame] = TypeMark(frame=frame, shot_type=row["type"] or None)
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
    is_hit_frame: bool = False,
    paused: bool = False,
    total_context: tuple[int, int] | None = None,
) -> None:
    height, width = canvas.shape[:2]
    mark = marks[index]
    labelled = sum(1 for m in marks if m.shot_type)

    # ---- top bar: where we are and what this hit is
    _band(canvas, 0, 92)
    _text(canvas, f"Golpe {index + 1} / {len(marks)}", (32, 40), 0.85, _INK, 2)
    progress = f"{labelled} etiquetados · {len(marks) - labelled} pendientes en este rally"
    if total_context is not None:  # how this rally sits in the whole dataset
        done_all, all_hits = total_context
        progress += f"   |   {done_all + labelled} / {all_hits} en total"
    _text(canvas, progress, (32, 72), 0.55, _MUTED)

    if mark.shot_type:
        color = TYPE_COLORS.get(mark.shot_type, _INK)
        label = mark.shot_type
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
        "z deshacer   <- -> golpe   SPACE pausa   a/d frame   r repetir   q guardar y salir",
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
    if is_hit_frame:
        _text(canvas, "IMPACTO", (width // 2 - 60, 130), 0.7, _ACCENT, 2)
    if paused:
        _text(canvas, "PAUSA", (32, 130), 0.6, _MUTED, 2)
    if saved:
        _text(canvas, "guardado", (width - 150, 130), 0.6, (140, 220, 140), 2)


def _load_clip(
    capture: cv2.VideoCapture, hit_frame: int, half: int
) -> tuple[list[ImageArray], int]:
    """Decode the frames around a hit into memory, plus the hit's index in them.

    Read sequentially from the clip's start: seeking per displayed frame makes
    the loop stutter, and a clip is small enough to hold in memory.
    """
    first = max(hit_frame - half, 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[ImageArray] = []
    for _ in range(half * 2 + 1):
        ok, image = capture.read()
        if not ok:
            break
        # Scale 1080p down to fit on screen: the window is AUTOSIZE (resizable
        # windows render blank under WSLg), so the frame must arrive at the size
        # it will be shown at.
        if image.shape[1] > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / image.shape[1]
            image = cv2.resize(image, (DISPLAY_WIDTH, round(image.shape[0] * scale)))
        frames.append(cast(ImageArray, image))
    return frames, hit_frame - first


def ensure_qt_fonts() -> None:
    """Give OpenCV's bundled Qt a font directory, or its windows render blank.

    OpenCV ships a Qt build that looks for fonts under `cv2/qt/fonts`, but no
    longer bundles any. Without them Qt floods stderr with QFontDatabase
    warnings and the window comes up empty — which on WSL looks exactly like
    "the GUI does not work". Symlinking the system DejaVu fonts fixes it.
    """
    fonts_dir = Path(cv2.__file__).parent / "qt" / "fonts"
    if fonts_dir.is_dir() and any(fonts_dir.glob("*.ttf")):
        return
    system_fonts = Path("/usr/share/fonts/truetype/dejavu")
    if not system_fonts.is_dir():
        return
    fonts_dir.mkdir(parents=True, exist_ok=True)
    for font in system_fonts.glob("*.ttf"):
        target = fonts_dir / font.name
        if not target.exists():
            target.symlink_to(font)


def _window_closed(window: str) -> bool:
    """Whether the user closed the window with the X.

    Qt raises "NULL guiReceiver" instead of returning a value when its window is
    already gone, so the query has to be guarded — an unguarded call crashes the
    annotator (and loses the session) exactly when the user closes the window.
    """
    try:
        return bool(cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1)
    except cv2.error:
        return True


def annotate_types(
    video_path: Path,
    hit_frames: list[int],
    out_csv: Path,
    window: str = "Tipo de golpe",
    clip_half: int = CLIP_HALF_FRAMES,
    total_context: tuple[int, int] | None = None,
) -> None:
    """Run the labelling loop over pre-located hits.

    Each hit plays as a looping clip rather than a still frame: a stroke type is
    a movement (wind-up, contact, follow-through) and cannot reliably be told
    from one image — by a human or, as it turns out, by the model, which sees a
    whole window too (ADR-0016 F).
    """
    ensure_qt_fonts()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    existing = load_marks(out_csv)
    marks = [existing.get(f, TypeMark(frame=f)) for f in hit_frames]
    # resume on the first unlabelled hit
    index = next((i for i, m in enumerate(marks) if not m.shot_type), 0)
    history: list[tuple[int, str | None]] = []

    # AUTOSIZE + an immediate first paint: under WSLg a window that is created
    # but not shown anything for a few seconds (decoding the first clip takes
    # that long) comes up blank, present in the taskbar but never rendered.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    splash = np.zeros((360, 640, 3), dtype=np.uint8)
    _text(splash, "cargando el primer golpe...", (60, 190), 0.8, _INK, 2)
    cv2.imshow(window, splash)
    cv2.waitKey(1)

    clip, hit_at = _load_clip(capture, marks[index].frame, clip_half)
    cursor = 0
    paused = False
    saved_flash = 0
    delay = max(int(1000 / fps), 1)  # real time

    while True:
        if clip:
            image = clip[min(cursor, len(clip) - 1)]
        else:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas = image.copy()
        _draw_hud(
            canvas,
            marks,
            index,
            marks[index].frame - hit_at + cursor,
            fps,
            saved_flash > 0,
            is_hit_frame=cursor == hit_at,
            paused=paused,
            total_context=total_context,
        )
        saved_flash = max(saved_flash - 1, 0)
        cv2.imshow(window, canvas)
        if not paused and clip:
            cursor = (cursor + 1) % len(clip)  # loop the clip continuously

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key == 255:  # no key pressed: keep looping unless the window is gone
            if _window_closed(window):
                break
            continue

        def go_to(new_index: int) -> None:
            nonlocal index, clip, hit_at, cursor, paused
            index = max(0, min(new_index, len(marks) - 1))
            clip, hit_at = _load_clip(capture, marks[index].frame, clip_half)
            cursor = 0
            paused = False

        if key in SHOT_TYPES:
            history.append((index, marks[index].shot_type))
            marks[index].shot_type = SHOT_TYPES[key]
            save_marks(marks, out_csv)
            saved_flash = 8
            if index < len(marks) - 1:  # advance automatically: keeps the rhythm
                go_to(index + 1)
        elif key == ord("z") and history:
            i, shot_type = history.pop()
            marks[i].shot_type = shot_type
            save_marks(marks, out_csv)
            go_to(i)
        elif key in (81, ord(",")):  # left arrow
            go_to(index - 1)
        elif key in (83, ord(".")):  # right arrow
            go_to(index + 1)
        elif key == ord(" "):
            paused = not paused
        elif key == ord("a"):  # step back one frame (implies pause)
            paused = True
            cursor = max(cursor - 1, 0)
        elif key == ord("d"):
            paused = True
            cursor = min(cursor + 1, len(clip) - 1 if clip else 0)
        elif key == ord("r"):  # replay from the start of the clip
            cursor, paused = 0, False
        elif key == ord("q"):
            save_marks(marks, out_csv)
            break
        elif key == 27:  # ESC
            break

    capture.release()
    cv2.destroyAllWindows()

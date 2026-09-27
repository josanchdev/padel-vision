"""Blind ground truth for whole rallies: the external evaluation's labels.

Footage from outside CVSPORTS has no ground truth, so Jorge makes it: he watches
each rally and marks every hit — the frame of contact, the player who hit it (a
click on him) and its type.

BLIND BY CONSTRUCTION. This tool never loads anything the system produced: no
audio candidates, no ball, no skeletons, no player numbers. Seeing detections
while labelling would bias exactly what is being measured — a hit the audio
missed would never be shown, so it would never be labelled, and recall would
come out inflated. The older `shot_annotator` can overlay the ball and jump
between audio candidates, which is why this is a separate tool rather than a
mode of that one: there is no switch to forget.

The player is a CLICK, not a number. A number would have to be the system's own
J1-J4, itself an output under test: if the system swapped two players mid-rally,
the label would silently follow the swap. A click is independent of all that;
the evaluation matches it to whichever detected player stands there.

Keys (the types as in the CVSPORTS labelling, so the habit carries over):
    click on the hitter at the contact frame, then
    1 Saque   2 Derecha   3 Reves   4 Remate   o  otro golpe (bandeja, dejada...)
    SPACE  play / pause, real speed         a / d   one frame back / forward
    <- / ->  one second back / forward      b / n   previous / next mark
    click on the bottom bar: jump there
    x  delete the mark on this frame        z  undo
    q / ESC  quit — every change is already saved
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from padel_cv.shot_type_annotator import (
    SHOT_TYPES,
    TYPE_COLORS,
    ImageArray,
    _band,
    _text,
    _window_closed,
    ensure_qt_fonts,
)

DISPLAY_WIDTH = 1280
"""Frames are shown at this width (1080p does not fit on screen)."""

TIMELINE_HEIGHT = 64
"""The bottom strip: the rally's timeline, clickable to jump."""

SPANISH = {
    "Serve": "Saque",
    "Forehand": "Derecha",
    "Backhand": "Reves",
    "Smash": "Remate",
    "Other": "Otro",
}
_ACCENT = (235, 190, 80)
_INK = (245, 245, 245)
_MUTED = (165, 165, 165)
_PENDING = (80, 230, 255)


@dataclass(frozen=True)
class HitMark:
    """One hit as a human saw it: when, where the hitter stood, and what it was."""

    frame: int
    x_px: float
    y_px: float
    """Click on the hitter, in the ORIGINAL video's pixels (not the display's)."""
    shot_type: str


def save_hit_marks(marks: list[HitMark], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "x_px", "y_px", "type"])
        for mark in sorted(marks, key=lambda m: m.frame):
            writer.writerow([mark.frame, f"{mark.x_px:.1f}", f"{mark.y_px:.1f}", mark.shot_type])


def load_hit_marks(path: Path) -> dict[int, HitMark]:
    if not path.exists():
        return {}
    out: dict[int, HitMark] = {}
    for row in csv.DictReader(path.open(), delimiter=";"):
        frame = int(row["frame"])
        out[frame] = HitMark(frame, float(row["x_px"]), float(row["y_px"]), row["type"])
    return out


class MarkBook:
    """One rally's marks, written to disk on every change (WSL crashes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.marks = load_hit_marks(path)
        self._history: list[tuple[int, HitMark | None]] = []

    def put(self, mark: HitMark) -> None:
        """Add a mark, replacing any already on that frame."""
        self._history.append((mark.frame, self.marks.get(mark.frame)))
        self.marks[mark.frame] = mark
        self._save()

    def delete(self, frame: int) -> bool:
        if frame not in self.marks:
            return False
        self._history.append((frame, self.marks.pop(frame)))
        self._save()
        return True

    def undo(self) -> int | None:
        """Revert the last put/delete; returns the frame it touched."""
        if not self._history:
            return None
        frame, previous = self._history.pop()
        if previous is None:
            self.marks.pop(frame, None)
        else:
            self.marks[frame] = previous
        self._save()
        return frame

    def neighbour(self, frame: int, forward: bool) -> int | None:
        """The closest mark strictly after (or before) `frame`."""
        frames = sorted(self.marks)
        if forward:
            return next((f for f in frames if f > frame), None)
        return next((f for f in reversed(frames) if f < frame), None)

    def _save(self) -> None:
        save_hit_marks(list(self.marks.values()), self.path)


class FrameCache:
    """Random access to a video's frames, decoded once and kept as JPEG in memory.

    Frames are read strictly in order, so a frame's index is its position in the
    stream and never an estimate. Seeking an H.264 file can land a frame or two
    off, which is not acceptable when the label IS the contact frame. Each frame
    is kept as a display-size JPEG (~150 KB): stepping backwards is instant and a
    two-minute rally costs a few hundred MB.
    """

    def __init__(self, video: Path, display_width: int = DISPLAY_WIDTH) -> None:
        self._capture = cv2.VideoCapture(str(video))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(display_width / width, 1.0)
        self.display_size = (round(width * scale), round(height * scale))
        self.to_original = (width / self.display_size[0], height / self.display_size[1])
        self._expected = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._jpegs: list[bytes] = []
        self._newest: ImageArray | None = None
        self.exhausted = False

    @property
    def decoded(self) -> int:
        return len(self._jpegs)

    @property
    def n_frames(self) -> int:
        """Exact once the video has been read to the end, the container's estimate before."""
        return self.decoded if self.exhausted else max(self._expected, self.decoded)

    def get(self, index: int) -> ImageArray | None:
        """Frame `index` at display size, or None past the end of the video."""
        while self.decoded <= index and not self.exhausted:
            self._decode_next()
        if index < 0 or index >= self.decoded:
            return None
        if index == self.decoded - 1 and self._newest is not None:
            return self._newest.copy()
        buffer = np.frombuffer(self._jpegs[index], dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return None if decoded is None else decoded.astype(np.uint8)

    def release(self) -> None:
        self._capture.release()

    def _decode_next(self) -> None:
        ok, image = self._capture.read()
        if not ok:
            self.exhausted = True
            self._capture.release()
            return
        small = cv2.resize(image, self.display_size, interpolation=cv2.INTER_AREA)
        _, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._jpegs.append(encoded.tobytes())
        self._newest = small.astype(np.uint8)


def frame_at(x: int, width: int, n_frames: int) -> int:
    """Timeline position -> frame."""
    fraction = min(max(x / max(width - 1, 1), 0.0), 1.0)
    return round(fraction * max(n_frames - 1, 0))


def x_of(frame: int, width: int, n_frames: int) -> int:
    """Frame -> timeline position."""
    return round(frame / max(n_frames - 1, 1) * (width - 1))


@dataclass
class _Session:
    cursor: int = 0
    playing: bool = False
    pending: tuple[int, float, float] | None = None
    """(frame, x, y) of a click waiting for its type, in original pixels."""
    message: str = ""
    message_until: float = 0.0
    seek_to: int | None = None

    def say(self, text: str, seconds: float = 1.6) -> None:
        self.message, self.message_until = text, time.perf_counter() + seconds


def _draw(canvas: ImageArray, session: _Session, book: MarkBook, frames: FrameCache) -> None:
    height, width = canvas.shape[:2]
    sx, sy = frames.to_original

    # ---- a mark on this very frame: Jorge's own label, shown back to him
    mark = book.marks.get(session.cursor)
    if mark is not None:
        centre = (round(mark.x_px / sx), round(mark.y_px / sy))
        colour = TYPE_COLORS.get(mark.shot_type, _INK)
        cv2.circle(canvas, centre, 22, colour, 3, cv2.LINE_AA)
        _text(
            canvas,
            SPANISH.get(mark.shot_type, mark.shot_type),
            (centre[0] + 28, centre[1]),
            0.7,
            colour,
            2,
        )

    # ---- a click waiting for its type
    if session.pending is not None and session.pending[0] == session.cursor:
        centre = (round(session.pending[1] / sx), round(session.pending[2] / sy))
        cv2.drawMarker(canvas, centre, _PENDING, cv2.MARKER_CROSS, 36, 2, cv2.LINE_AA)
        _band(canvas, 96, 138)
        _text(
            canvas,
            "tipo:  1 Saque   2 Derecha   3 Reves   4 Remate   o Otro",
            (32, 126),
            0.7,
            _PENDING,
            2,
        )

    # ---- top bar
    _band(canvas, 0, 88)
    time_s = session.cursor / frames.fps
    state = "REPRODUCIENDO" if session.playing else "PAUSA"
    _text(
        canvas,
        f"{state}   frame {session.cursor}   {int(time_s // 60)}:{time_s % 60:05.2f}",
        (32, 38),
        0.75,
        _INK,
        2,
    )
    counts = {name: 0 for name in SPANISH}
    for m in book.marks.values():
        counts[m.shot_type] = counts.get(m.shot_type, 0) + 1
    summary = "   ".join(f"{SPANISH[k]} {v}" for k, v in counts.items() if v)
    _text(canvas, f"{len(book.marks)} golpes marcados   {summary}", (32, 70), 0.55, _MUTED)
    if time.perf_counter() < session.message_until:
        size = cv2.getTextSize(session.message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        _text(canvas, session.message, (width - size[0] - 32, 38), 0.7, _ACCENT, 2)

    # ---- bottom: timeline (clickable) and the keys
    top = height - TIMELINE_HEIGHT
    _band(canvas, top, height, alpha=0.8)
    bar_y = top + 18
    n = frames.n_frames
    cv2.line(canvas, (0, bar_y), (width - 1, bar_y), (70, 70, 70), 6)
    # decoded so far: a far jump forward has to read up to there first
    cv2.line(canvas, (0, bar_y), (x_of(frames.decoded, width, n), bar_y), (110, 110, 110), 6)
    for m in book.marks.values():
        x = x_of(m.frame, width, n)
        cv2.line(canvas, (x, bar_y - 10), (x, bar_y + 10), TYPE_COLORS.get(m.shot_type, _INK), 3)
    x = x_of(session.cursor, width, n)
    cv2.drawMarker(canvas, (x, bar_y - 12), _ACCENT, cv2.MARKER_TRIANGLE_DOWN, 14, 2, cv2.LINE_AA)
    _text(
        canvas,
        "clic en el jugador + 1-4/o   SPACE play   a/d frame   <-/-> 1 s   b/n marca   "
        "x borrar   z deshacer   q salir",
        (24, height - 14),
        0.5,
        _MUTED,
    )


def annotate_blind(video: Path, out_csv: Path, window: str = "Etiquetado a ciegas") -> None:
    """Watch a rally and mark every hit: contact frame, hitter (click) and type."""
    ensure_qt_fonts()
    # AUTOSIZE + an immediate first paint: under WSLg a window left empty while
    # the first frames decode comes up blank, present in the taskbar only.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    splash = np.zeros((360, 640, 3), dtype=np.uint8)
    _text(splash, "abriendo el video...", (60, 190), 0.8, _INK, 2)
    cv2.imshow(window, splash)
    cv2.waitKey(1)

    frames = FrameCache(video)
    book = MarkBook(out_csv)
    session = _Session()
    display_height = frames.display_size[1]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y >= display_height - TIMELINE_HEIGHT:
            session.seek_to = frame_at(x, frames.display_size[0], frames.n_frames)
            return
        sx, sy = frames.to_original
        session.playing = False
        session.pending = (session.cursor, x * sx, y * sy)

    cv2.setMouseCallback(window, on_mouse)
    next_tick = time.perf_counter()
    step = max(round(frames.fps), 1)

    while True:
        if session.seek_to is not None:
            session.cursor, session.seek_to = session.seek_to, None
        image = frames.get(session.cursor)
        if image is None:  # ran past the end
            session.cursor = max(frames.decoded - 1, 0)
            session.playing = False
            image = frames.get(session.cursor)
            if image is None:
                break
        _draw(image, session, book, frames)
        cv2.imshow(window, image)

        if session.playing:
            next_tick += 1 / frames.fps
            lag = next_tick - time.perf_counter()
            if lag < -0.25:  # fell behind (a slow decode): resync rather than rush
                next_tick = time.perf_counter()
            wait = max(int(lag * 1000), 1)
        else:
            wait = 30
        key = cv2.waitKey(wait) & 0xFF

        if key == 255:
            if _window_closed(window):
                break
        elif key in SHOT_TYPES:
            pending = session.pending
            if pending is None or pending[0] != session.cursor:
                session.say("primero haz clic en el jugador que golpea")
            else:
                book.put(HitMark(session.cursor, pending[1], pending[2], SHOT_TYPES[key]))
                session.pending = None
                session.say(f"guardado: {SPANISH[SHOT_TYPES[key]]}")
        elif key == ord(" "):
            session.playing = not session.playing
            session.pending = None
            next_tick = time.perf_counter()
        elif key == ord("a"):
            session.playing, session.cursor = False, max(session.cursor - 1, 0)
        elif key == ord("d"):
            session.playing, session.cursor = False, session.cursor + 1
        elif key in (81, ord(",")):  # left arrow
            session.cursor = max(session.cursor - step, 0)
        elif key in (83, ord(".")):  # right arrow
            session.cursor += step
        elif key in (ord("b"), ord("n")):
            target = book.neighbour(session.cursor, forward=key == ord("n"))
            if target is not None:
                session.playing, session.cursor = False, target
        elif key == ord("x"):
            if book.delete(session.cursor):
                session.say("marca borrada")
        elif key == ord("z"):
            frame = book.undo()
            if frame is not None:
                session.playing, session.cursor = False, frame
                session.say("deshecho")
        elif key in (ord("q"), 27):
            break

        if session.playing and key == 255:
            session.cursor += 1

    frames.release()
    cv2.destroyAllWindows()

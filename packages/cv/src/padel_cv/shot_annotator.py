"""Quick-mark shot annotator (ADR-0014): scrub a video, tap a key per shot.

The bottleneck for the shot models is DATA, not architecture (measured: heuristic
15-52% recall, learned detector 29% sliding over real video). Annotating keypoints
per shot is far too slow to reach thousands. Instead: watch the video, pause on
each shot, press one key for the stroke type. One mark per shot trains BOTH the
detector (the frame = where a shot is) and the classifier (the type).

Navigation is by TIME, not frames, because match videos run at 60fps and 66 min:
stepping one frame is hopeless. The ball is drawn on top (if a detections file is
given) so obvious errors are visible; poses are not drawn — the job is marking
shots, not fixing keypoints.

Output CSV columns (ADR-0014): frame, type, from_wall, player.
Keys:
  SPACE   play / pause              + / -   play faster / slower
  <- / -> back / forward 1 second   Up/Down back / forward 5 seconds
  , / .   step 1 frame (fine tune)  b / n   jump to prev / next marked shot
  1..5    mark shot at current frame (Serve/Forehand/Backhand/Lob/Smash)
  w       toggle 'from wall' on last mark      z   undo last mark
  LEFT-CLICK (while paused)  move/set the ball on this frame (fix the detector)
  s       save now
  q       quit AND save             ESC     quit WITHOUT saving (asks)

Resumes where you left off (position saved next to the CSV). Ball corrections are
saved to <csv>.ball.json; the models use them at shot frames (ADR-0014).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]

# Stroke types by key (ADR-0014): gesture only; wall is a separate flag.
SHOT_TYPES = {
    ord("1"): "Serve",
    ord("2"): "Forehand",
    ord("3"): "Backhand",
    ord("4"): "Lob",
    ord("5"): "Smash",
}
_PLAY_SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]  # cycled by + / -
_BALL_COLOR = (0, 255, 255)
_HELP_LINES = [
    "SPACE play/pause  +/- speed   <-/-> 1s   Up/Dn 5s   ,/. 1 frame",
    "1 Serve 2 Forehand 3 Backhand 4 Lob 5 Smash   w wall   z undo",
    "b/n prev/next shot   CLICK=move ball   s save   q save+quit   ESC no-save",
]


@dataclass
class ShotMark:
    frame: int
    type: str
    from_wall: bool
    player: int  # -1 = unknown (not assigned in this pass)


def _load_ball(path: Path | None) -> dict[int, tuple[float, float]]:
    """frame -> (x, y) from a detections JSON (COCO ball boxes or [{frame,x,y}])."""
    import json

    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, tuple[float, float]] = {}
    if isinstance(data, dict) and "annotations" in data:  # COCO ball boxes
        frame_of = {img["id"]: _coco_frame(img["file_name"]) for img in data["images"]}
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            out.setdefault(frame_of[ann["image_id"]], (x + w / 2, y + h / 2))
    elif isinstance(data, list):
        for d in data:
            out[int(d["frame"])] = (float(d["x"]), float(d["y"]))
    return out


def _coco_frame(file_name: str) -> int:
    return int(Path(file_name).stem.split("_")[-1])


def _draw_ball(canvas: ImageArray, xy: tuple[float, float] | None, corrected: bool) -> None:
    if xy is None:
        return
    x, y = int(xy[0]), int(xy[1])
    color = (0, 0, 255) if corrected else _BALL_COLOR  # red when hand-corrected
    cv2.circle(canvas, (x, y), 8, color, 2)
    cv2.circle(canvas, (x, y), 1, color, -1)


def ball_corrections_path(csv_path: Path) -> Path:
    """Sidecar JSON for hand-corrected ball positions at shot frames."""
    return csv_path.with_suffix(csv_path.suffix + ".ball.json")


def load_ball_corrections(csv_path: Path) -> dict[int, tuple[float, float]]:
    import json

    p = ball_corrections_path(csv_path)
    if not p.exists():
        return {}
    return {int(k): (float(v[0]), float(v[1])) for k, v in json.loads(p.read_text()).items()}


def save_ball_corrections(csv_path: Path, corr: dict[int, tuple[float, float]]) -> None:
    import json

    ball_corrections_path(csv_path).write_text(
        json.dumps({str(k): [v[0], v[1]] for k, v in corr.items()})
    )


def _fmt_time(frame: int, fps: float) -> str:
    s = frame / fps if fps else 0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def _overlay(
    canvas: ImageArray,
    frame_idx: int,
    total: int,
    fps: float,
    marks: list[ShotMark],
    paused: bool,
    speed: float,
    ball_here: bool,
) -> None:
    """Draw HUD: time/frame, shot count, speed, last mark, key help."""
    h = canvas.shape[0]
    play = "PAUSED" if paused else f"PLAY {speed:g}x"
    status = f"{_fmt_time(frame_idx, fps)}  f{frame_idx}/{total - 1}  shots:{len(marks)}  {play}"
    _text(canvas, status, (12, 30), 0.8, (255, 255, 255))
    if not ball_here:
        _text(canvas, "no ball this frame", (12, 58), 0.6, (0, 165, 255))
    if marks:
        last = marks[-1]
        wall = " [WALL]" if last.from_wall else ""
        _text(canvas, f"last: {last.type}{wall} @ f{last.frame}", (12, 84), 0.7, (0, 220, 220))
    for i, line in enumerate(_HELP_LINES):
        y = h - 14 - (len(_HELP_LINES) - 1 - i) * 24
        _text(canvas, line, (12, y), 0.55, (240, 240, 240))
    # Green dot top-right when a marked shot is within ~1s of the current frame.
    if any(abs(m.frame - frame_idx) <= fps for m in marks):
        cv2.circle(canvas, (canvas.shape[1] - 30, 30), 11, (0, 220, 0), -1)


def _text(
    canvas: ImageArray, s: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]
) -> None:
    """White/coloured text with a black outline so it reads over any frame."""
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def save_marks(marks: list[ShotMark], csv_path: Path) -> None:
    """Write marks to CSV (sorted by frame), columns per ADR-0014."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["frame", "type", "from_wall", "player"])
        for m in sorted(marks, key=lambda m: m.frame):
            w.writerow([m.frame, m.type, int(m.from_wall), m.player])


def load_marks(csv_path: Path) -> list[ShotMark]:
    """Resume a previous session's CSV if present."""
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open(), delimiter=";"))
    return [
        ShotMark(int(r["frame"]), r["type"], bool(int(r["from_wall"])), int(r["player"]))
        for r in rows
    ]


def _pos_path(csv_path: Path) -> Path:
    """Sidecar file storing where the annotator was last, to resume there."""
    return csv_path.with_suffix(csv_path.suffix + ".pos")


def _load_resume_frame(csv_path: Path, marks: list[ShotMark]) -> int:
    """Where to open at: the saved position, else just after the last marked shot."""
    pos = _pos_path(csv_path)
    if pos.exists():
        try:
            return max(0, int(pos.read_text().strip()))
        except ValueError:
            pass
    return max((m.frame for m in marks), default=0)


def _save_pos(csv_path: Path, frame: int) -> None:
    _pos_path(csv_path).write_text(str(frame))


def _next_shot(frame: int, marks: list[ShotMark], forward: bool) -> int | None:
    """Frame of the nearest marked shot after/before `frame`, or None."""
    frames = sorted(m.frame for m in marks)
    if forward:
        later = [f for f in frames if f > frame]
        return later[0] if later else None
    earlier = [f for f in frames if f < frame]
    return earlier[-1] if earlier else None


_WINDOW = "padel shot annotator"


def annotate(
    video_path: Path,
    csv_out: Path,
    ball_json: Path | None = None,
    start_frame: int = 0,
) -> list[ShotMark]:
    """Run the interactive annotator; returns marks (saved to csv_out on q/s)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    sec = round(fps)  # frames per second, for 1-second jumps
    ball = _load_ball(ball_json)
    marks = load_marks(csv_out)  # resume if the CSV already exists
    ball_fix = load_ball_corrections(csv_out)  # hand-corrected ball positions

    # Resume where we left off: the saved position, else after the last shot, else
    # the caller's start_frame.
    resumed = _load_resume_frame(csv_out, marks)
    idx = max(0, min(resumed or start_frame, total - 1))
    paused = True
    speed_i = 1  # index into _PLAY_SPEEDS (1.0x)
    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    # Left-click while paused moves the ball for the CURRENT frame (correcting the
    # detector at a shot). The callback writes into `click`, the loop applies it.
    click: dict[str, tuple[int, int] | None] = {"xy": None}

    def _on_mouse(event: int, mx: int, my: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (mx, my)

    cv2.setMouseCallback(_WINDOW, _on_mouse)

    # Seek is expensive at 1080p, so only seek when idx JUMPS; during play we read
    # sequentially (fast). `pos` tracks where the decoder actually is.
    pos = -1

    while True:
        if idx != pos:  # a jump (arrows, shot navigation) -> seek once
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = capture.read()
        if not ok:
            idx = pos = max(0, idx - 1)
            paused = True
            continue
        pos = idx + 1  # read() advanced the decoder to the next frame

        # Apply a pending ball correction to THIS frame (paused only).
        if paused and click["xy"] is not None:
            ball_fix[idx] = (float(click["xy"][0]), float(click["xy"][1]))
            save_ball_corrections(csv_out, ball_fix)
        click["xy"] = None

        ball_xy = ball_fix.get(idx, ball.get(idx))
        canvas: ImageArray = image.copy().astype(np.uint8)
        _draw_ball(canvas, ball_xy, corrected=idx in ball_fix)
        _overlay(canvas, idx, total, fps, marks, paused, _PLAY_SPEEDS[speed_i], ball_xy is not None)
        cv2.imshow(_WINDOW, canvas)

        delay = 1 if paused else max(1, int(1000 / (fps * _PLAY_SPEEDS[speed_i])))
        key = cv2.waitKey(delay) & 0xFF

        # Window closed with the X button -> save and exit.
        if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("q"):
            break  # save + quit
        elif key == 27:  # ESC: quit WITHOUT saving (confirm on the frame)
            if _confirm_discard(canvas):
                capture.release()
                cv2.destroyAllWindows()
                return marks
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            speed_i = min(len(_PLAY_SPEEDS) - 1, speed_i + 1)
        elif key == ord("-"):
            speed_i = max(0, speed_i - 1)
        elif key == 81:  # left arrow: -1 second
            idx = max(0, idx - sec)
            paused = True
        elif key == 83:  # right arrow: +1 second
            idx = min(total - 1, idx + sec)
            paused = True
        elif key == 82:  # up arrow: +5 seconds
            idx = min(total - 1, idx + 5 * sec)
            paused = True
        elif key == 84:  # down arrow: -5 seconds
            idx = max(0, idx - 5 * sec)
            paused = True
        elif key == ord(","):  # fine tune -1 frame
            idx = max(0, idx - 1)
            paused = True
        elif key == ord("."):  # fine tune +1 frame
            idx = min(total - 1, idx + 1)
            paused = True
        elif key == ord("n"):  # jump to next marked shot
            nxt = _next_shot(idx, marks, forward=True)
            if nxt is not None:
                idx, paused = nxt, True
        elif key == ord("b"):  # jump to previous marked shot
            prv = _next_shot(idx, marks, forward=False)
            if prv is not None:
                idx, paused = prv, True
        elif key in SHOT_TYPES:
            marks.append(ShotMark(idx, SHOT_TYPES[key], from_wall=False, player=-1))
            save_marks(marks, csv_out)  # autosave on every mark
        elif key == ord("w") and marks:
            marks[-1].from_wall = not marks[-1].from_wall
            save_marks(marks, csv_out)
        elif key == ord("z") and marks:
            marks.pop()
            save_marks(marks, csv_out)
        elif key == ord("s"):
            save_marks(marks, csv_out)
            _save_pos(csv_out, idx)
        elif not paused:
            idx = min(total - 1, idx + 1)
            if idx == total - 1:
                paused = True

    capture.release()
    cv2.destroyAllWindows()
    save_marks(marks, csv_out)
    _save_pos(csv_out, idx)  # remember where we stopped, to resume next time
    return marks


def _confirm_discard(base: ImageArray) -> bool:
    """Ask on-screen whether to quit without saving; returns True to discard."""
    canvas = base.copy()
    _text(canvas, "Quit WITHOUT saving? y = yes, any other = cancel", (12, 120), 0.8, (0, 0, 255))
    cv2.imshow(_WINDOW, canvas)
    return (cv2.waitKey(0) & 0xFF) == ord("y")


def _marks_summary(marks: list[ShotMark]) -> dict[str, int]:
    """Count per type (for a quick post-session print)."""
    out: dict[str, int] = {}
    for m in marks:
        out[m.type] = out.get(m.type, 0) + 1
    return out


__all__ = ["SHOT_TYPES", "ShotMark", "_marks_summary", "annotate", "load_marks", "save_marks"]

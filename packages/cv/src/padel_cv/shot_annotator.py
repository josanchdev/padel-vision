"""Quick-mark shot annotator (ADR-0014): scrub a video, tap a key per shot.

The bottleneck for the shot models is DATA, not architecture (measured: heuristic
15-52% recall, learned detector 29% sliding over real video). Annotating keypoints
per shot is far too slow to reach thousands. Instead: watch the video, pause on
each shot, press one key for the stroke type. One mark per shot trains BOTH the
detector (the frame = where a shot is) and the classifier (the type).

This is a purpose-built OpenCV player, not CVAT: CVAT moves boxes/keypoints
(slow); this is mark-and-classify (~5-10s per shot). Pose and ball, if a
detections file is given, are drawn on top so obvious errors are visible — but the
job is marking shots, not fixing keypoints.

Output CSV columns (ADR-0014): frame, type, from_wall, player.
Keys:
  SPACE  play / pause            , / .   jump back / forward 10 frames
  <- / ->  step 1 frame          [ / ]   jump 100 frames
  1..5   mark shot of this type at the current frame (Serve/Forehand/Backhand/Lob/Smash)
  w      toggle 'from wall' on the last mark
  z      undo last mark          s   save CSV        q   quit (autosaves)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
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

_BALL_COLOR = (0, 255, 255)
_HELP_LINES = [
    "SPACE play/pause  <-/-> step  ,/. +-10  [/] +-100",
    "1 Serve  2 Forehand  3 Backhand  4 Lob  5 Smash",
    "w from-wall(last)  z undo  s save  q quit",
]


@dataclass
class ShotMark:
    frame: int
    type: str
    from_wall: bool
    player: int  # -1 = unknown (not assigned in this pass)


def _load_ball(path: Path | None) -> dict[int, tuple[float, float]]:
    """frame -> (x, y) from a detections JSON (list of {frame,x,y} or COCO)."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, tuple[float, float]] = {}
    if isinstance(data, dict) and "annotations" in data:  # COCO ball boxes
        frame_of = {img["id"]: _coco_frame(img["file_name"]) for img in data["images"]}
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            out.setdefault(frame_of[ann["image_id"]], (x + w / 2, y + h / 2))
    elif isinstance(data, list):  # simple [{frame,x,y}, ...]
        for d in data:
            out[int(d["frame"])] = (float(d["x"]), float(d["y"]))
    return out


def _coco_frame(file_name: str) -> int:
    return int(Path(file_name).stem.split("_")[-1])


def _draw_ball(canvas: ImageArray, xy: tuple[float, float] | None) -> None:
    if xy is None:
        return
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(canvas, (x, y), 7, _BALL_COLOR, 2)
    cv2.circle(canvas, (x, y), 1, _BALL_COLOR, -1)


def _overlay(
    canvas: ImageArray,
    frame_idx: int,
    total: int,
    marks: list[ShotMark],
    paused: bool,
) -> None:
    """Draw HUD: frame counter, shot count, last mark, and the key help."""
    h = canvas.shape[0]
    status = f"frame {frame_idx}/{total - 1}  shots {len(marks)}  {'PAUSED' if paused else 'PLAY'}"
    cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    if marks:
        last = marks[-1]
        wall = " [WALL]" if last.from_wall else ""
        txt = f"last: {last.type}{wall} @ {last.frame}"
        cv2.putText(canvas, txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
    for i, line in enumerate(_HELP_LINES):
        y = h - 14 - (len(_HELP_LINES) - 1 - i) * 24
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    # A marker line near shot frames so you can see what's already annotated.
    for m in marks:
        if abs(m.frame - frame_idx) <= 45:
            cv2.circle(canvas, (canvas.shape[1] - 30, 30), 10, (0, 220, 0), -1)
            break


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


def annotate(
    video_path: Path,
    csv_out: Path,
    ball_json: Path | None = None,
    start_frame: int = 0,
    fps_play: float = 30.0,
) -> list[ShotMark]:
    """Run the interactive annotator; returns the marks (also saved to csv_out)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ball = _load_ball(ball_json)
    marks = load_marks(csv_out)  # resume if the CSV already exists

    idx = max(0, min(start_frame, total - 1))
    paused = True
    play_delay = max(1, int(1000 / fps_play))
    window = "padel shot annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = capture.read()
        if not ok:
            idx = max(0, idx - 1)
            paused = True
            continue
        canvas: ImageArray = image.copy().astype(np.uint8)
        _draw_ball(canvas, ball.get(idx))
        _overlay(canvas, idx, total, marks, paused)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(0 if paused else play_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (81, ord(",")):  # left arrow (81) / comma
            idx = max(0, idx - (10 if key == ord(",") else 1))
            paused = True
        elif key in (83, ord(".")):  # right arrow (83) / period
            idx = min(total - 1, idx + (10 if key == ord(".") else 1))
            paused = True
        elif key == ord("["):
            idx = max(0, idx - 100)
            paused = True
        elif key == ord("]"):
            idx = min(total - 1, idx + 100)
            paused = True
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
        elif not paused:
            idx = min(total - 1, idx + 1)
            if idx == total - 1:
                paused = True

    capture.release()
    cv2.destroyAllWindows()
    save_marks(marks, csv_out)
    return marks


def _marks_summary(marks: list[ShotMark]) -> dict[str, int]:
    """Count per type (for a quick post-session print)."""
    out: dict[str, int] = {}
    for m in marks:
        out[m.type] = out.get(m.type, 0) + 1
    return out


__all__ = ["SHOT_TYPES", "ShotMark", "_marks_summary", "annotate", "load_marks", "save_marks"]


def _asdict_list(marks: list[ShotMark]) -> list[dict[str, object]]:
    return [asdict(m) for m in marks]

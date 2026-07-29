"""End-to-end evaluation of shot DETECTION and CLASSIFICATION against the
PadelTracker100 ground truth.

Motivation (29 jul 2026): validating on a real match suggested the shot pipeline
is weak on BOTH axes — it misses many real shots (low detection recall) and, of
the ones it catches, mislabels the type. Optimizing blind is pointless; this
harness measures the two axes SEPARATELY so we can see where the bleeding is.

Ground truth (`*_shots.csv`) marks `has_shot=1` over a BLOCK of frames per shot
(a shot lasts ~11-17 frames), not a single impact frame. So the truth is the set
of contiguous blocks; each block is ONE real shot, its impact taken at the block
centre. A predicted event matches a block if it lands within `tol` frames of that
centre (default ±3, the strict "did you time the impact right?" criterion).

We report, separately:
- DETECTION: recall = matched blocks / real blocks; precision = matched / emitted.
  This is the top of Jorge's "8-of-100" funnel — how many shots we even catch.
- CLASSIFICATION: of the matched shots, the fraction whose predicted type equals
  the GT block's type. This is the second stage of the funnel.
- END-TO-END: correct-type-and-detected / real shots. The honest headline number.

The detector here is the ADR-0012 method (ball direction-change + wrist near),
fed with GT pose+ball to isolate the ALGORITHM from our own detectors' noise.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from padel_cv.ball_events import BallSample, classify_events, detect_direction_changes


@dataclass(frozen=True)
class ShotBlock:
    """One ground-truth shot: a contiguous run of has_shot=1 frames."""

    start: int
    end: int  # inclusive
    category: str

    @property
    def centre(self) -> int:
        return (self.start + self.end) // 2


def load_shot_blocks(csv_path: Path) -> list[ShotBlock]:
    """Collapse the per-frame has_shot flags into one block per real shot."""
    rows = list(csv.DictReader(csv_path.open(), delimiter=";"))
    blocks: list[ShotBlock] = []
    run_start: int | None = None
    run_cats: list[str] = []
    for i, row in enumerate(rows):
        if row["has_shot"] == "1":
            if run_start is None:
                run_start = i
            run_cats.append(row["category"])
        elif run_start is not None:
            cat = Counter(run_cats).most_common(1)[0][0]
            blocks.append(ShotBlock(run_start, i - 1, cat))
            run_start, run_cats = None, []
    if run_start is not None:
        cat = Counter(run_cats).most_common(1)[0][0]
        blocks.append(ShotBlock(run_start, len(rows) - 1, cat))
    return blocks


def load_ball_track(ball_json: Path) -> list[BallSample]:
    """Per-frame ball centre from the GT COCO ball annotations (bbox centre)."""
    data = json.loads(ball_json.read_text())
    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in data["images"]}
    # A few frames carry 2-3 ball boxes (annotation noise / reflections); keep one
    # centre per frame so the trajectory stays single-valued.
    by_frame: dict[int, tuple[float, float]] = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        fi = frame_of[ann["image_id"]]
        by_frame.setdefault(fi, (x + w / 2, y + h / 2))
    return [BallSample(fi, cx, cy) for fi, (cx, cy) in sorted(by_frame.items())]


def load_wrists_and_torso(
    pose_json: Path,
) -> dict[int, list[tuple[list[tuple[float, float]], float]]]:
    """frame -> list of (wrists, torso_length) for every person in that frame.

    We keep every detected person (no player-id tracking): to measure detection
    recall we only need "is SOME wrist near the ball", which is what the detector
    does too. Head keypoints are unreliable in this dataset, but we only use
    wrists/shoulders/hips here, which are fine.
    """
    data = json.loads(pose_json.read_text())
    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in data["images"]}
    out: dict[int, list[tuple[list[tuple[float, float]], float]]] = {}
    for ann in data["annotations"]:
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        torso = _torso_length(kp)
        if torso is None:
            continue
        wrists = _wrists(kp)
        if not wrists:
            continue
        fi = frame_of[ann["image_id"]]
        out.setdefault(fi, []).append((wrists, torso))
    return out


# COCO-17 indices.
_L_SHO, _R_SHO, _L_HIP, _R_HIP = 5, 6, 11, 12
_L_WRI, _R_WRI = 9, 10
_MIN_CONF = 0.3


def _wrists(kp: np.ndarray) -> list[tuple[float, float]]:
    return [(float(kp[i, 0]), float(kp[i, 1])) for i in (_L_WRI, _R_WRI) if kp[i, 2] >= _MIN_CONF]


def _torso_length(kp: np.ndarray) -> float | None:
    parts = kp[[_L_SHO, _R_SHO, _L_HIP, _R_HIP]]
    if (parts[:, 2] < _MIN_CONF).any():
        return None
    shoulders = parts[:2, :2].mean(axis=0)
    hips = parts[2:, :2].mean(axis=0)
    length = float(np.linalg.norm(shoulders - hips))
    return length if length > 1.0 else None


def _frame_index(file_name: str) -> int:
    """'frame_000123.PNG' -> 123."""
    stem = Path(file_name).stem
    return int(stem.split("_")[-1])


@dataclass
class DetectionResult:
    real: int
    emitted: int
    matched: int  # emitted events that hit a real block
    covered: int  # distinct real blocks that got at least one event
    fp_bounce_hits: int = 0  # emitted BOUNCES that landed on a real shot block

    @property
    def recall(self) -> float:
        return self.covered / self.real if self.real else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.emitted if self.emitted else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_detection(
    ball: list[BallSample],
    wrists_by_frame: dict[int, list[tuple[list[tuple[float, float]], float]]],
    blocks: list[ShotBlock],
    *,
    tol: int = 3,
    min_turn_deg: float = 45.0,
    wrist_dist_frac: float = 1.2,
    lookahead: int = 3,
) -> DetectionResult:
    """Run the ADR-0012 detector on GT signals and match against GT blocks.

    A predicted SHOT event matches a block if it is within `tol` frames of the
    block's centre. Detection is scored on shot events only (bounces are the
    detector's other output; we track how often a bounce lands on a real shot as
    a separate confusion signal).
    """

    def wrist_near(frame_index: int, ball_xy: tuple[float, float]) -> int | None:
        best = float("inf")
        for wrists, torso in wrists_by_frame.get(frame_index, []):
            for wx, wy in wrists:
                d = float(np.hypot(wx - ball_xy[0], wy - ball_xy[1])) / torso
                best = min(best, d)
        # Player id is irrelevant for detection recall; return a sentinel id
        # whenever a wrist is close enough, else None (-> bounce).
        return 0 if best <= wrist_dist_frac else None

    raw = detect_direction_changes(ball, window=lookahead, min_turn_deg=min_turn_deg)
    events = classify_events(ball, raw, wrist_near)

    centres = np.array([b.centre for b in blocks])
    covered: set[int] = set()
    matched = 0
    emitted = 0
    bounce_hits = 0
    for ev in events:
        if ev.kind == "shot":
            emitted += 1
            hit = _nearest_block(ev.frame_index, centres, tol)
            if hit is not None:
                matched += 1
                covered.add(hit)
        elif _nearest_block(ev.frame_index, centres, tol) is not None:
            bounce_hits += 1
    return DetectionResult(
        real=len(blocks),
        emitted=emitted,
        matched=matched,
        covered=len(covered),
        fp_bounce_hits=bounce_hits,
    )


def _nearest_block(frame: int, centres: np.ndarray, tol: int) -> int | None:
    """Index of the GT block whose centre is within tol frames, else None."""
    if len(centres) == 0:
        return None
    idx = int(np.argmin(np.abs(centres - frame)))
    return idx if abs(int(centres[idx]) - frame) <= tol else None

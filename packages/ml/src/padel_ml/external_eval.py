"""Score the whole system against blind ground truth on external footage.

Ground truth comes from `padel_cv.blind_annotator`: for every hit, the contact
frame, a click on the hitter and the stroke type, labelled without seeing any
system output. The system runs end to end (`analyze_rally`, the same code as the
demo and the web) and is scored layer by layer:

    WHEN   detection: system hits paired with labelled ones within 250 ms — the
           collar of the audio detector's evaluation and of the paper.
    WHO    on paired hits, did the system pick the person Jorge clicked on? The
           click is resolved to whichever detected player stands there, so the
           score does not depend on the system keeping J1-J4 consistent. Team =
           same side of the net, measured on the court, not read off an ID.
    WHAT   on paired hits labelled with one of the four classes. Given twice:
           over every paired hit (errors of WHO propagate, as in real use) and
           only where WHO was right (the classifier on its own, comparable to
           the cross-tournament figure).
    END TO END   labelled hits that came out fully right: detected, right
           player, right type.

"Other" labels (bandeja, dejada, anything outside the four classes) count as
hits for WHEN and WHO but are left out of WHAT.
"""

from __future__ import annotations

import collections
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from padel_cv.blind_annotator import HitMark
from padel_cv.court import COURT_LENGTH_M
from padel_cv.pipeline import PoseDetection
from padel_ml.audio_train import match_events
from padel_ml.hit_assignment import MAX_POSE_GAP
from padel_ml.rally_analysis import RallyAnalysis
from padel_ml.shot_type_dataset import CLASSES

COLLAR_S = 0.25
"""Pairing tolerance, as in the audio evaluation and the paper."""

MAX_CLICK_DISTANCE = 0.5
"""A click farther than this from every player's box (in body heights) means the
hitter was not detected at all."""

FREEZE_MIN_FRAMES = 3
FREEZE_MAX_CHANGED_PX = 3
"""A frame is a repeat of the previous one when at most this many pixels change
by more than 25 levels (blurred, 640x360). Real motion changes hundreds."""


@dataclass(frozen=True)
class HitRow:
    """One labelled or detected hit, and how the system did on it."""

    clip: str
    truth_frame: int | None
    """None for a false detection (nothing labelled there)."""
    truth_type: str | None
    system_frame: int | None
    """None for a missed hit."""
    system_type: str | None
    true_player: int | None
    """System ID of the person Jorge clicked on; None if that person was not detected."""
    system_player: int | None
    player_ok: bool | None
    team_ok: bool | None
    type_ok: bool | None
    in_freeze: bool


def load_truth(path: Path) -> list[HitMark]:
    from padel_cv.blind_annotator import load_hit_marks

    return sorted(load_hit_marks(path).values(), key=lambda m: m.frame)


def frozen_frames(video: Path) -> set[int]:
    """Frames inside runs where the image stops moving (a broadcast freeze).

    A hit labelled there has no visible contact — for Jorge or for the system —
    so such hits are flagged and reported apart.
    """
    capture = cv2.VideoCapture(str(video))
    previous = None
    repeats: list[bool] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (640, 360))
        grey = cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)
        if previous is not None:
            changed = int((np.abs(grey - previous) > 25).sum())
            repeats.append(changed <= FREEZE_MAX_CHANGED_PX)
        previous = grey
    capture.release()
    frozen: set[int] = set()
    i = 0
    while i < len(repeats):
        if not repeats[i]:
            i += 1
            continue
        j = i
        while j < len(repeats) and repeats[j]:
            j += 1
        if j - i + 1 >= FREEZE_MIN_FRAMES:  # frames i..j are the same image
            frozen.update(range(i, j + 1))
        i = j
    return frozen


def _box_distance(pose: PoseDetection, x: float, y: float) -> float:
    """Distance from a point to a player's box, in his body heights (0 inside)."""
    x1, y1, x2, y2 = pose.bbox_xyxy
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy)) / max(y2 - y1, 1.0)


def clicked_player(mark: HitMark, players: dict[int, list[PoseDetection]]) -> PoseDetection | None:
    """The detected player Jorge clicked on, looking a few frames either side.

    The detector drops a player for a frame or two, the contact frame included,
    so a miss exactly there is bridged like the assignment itself does
    (`MAX_POSE_GAP`). Nearest frame first, then nearest box.
    """
    for offset in sorted(range(-MAX_POSE_GAP, MAX_POSE_GAP + 1), key=abs):
        candidates = players.get(mark.frame + offset, [])
        if not candidates:
            continue
        best = min(candidates, key=lambda p: _box_distance(p, mark.x_px, mark.y_px))
        if _box_distance(best, mark.x_px, mark.y_px) <= MAX_CLICK_DISTANCE:
            return best
    return None


def _side_of_net(pose: PoseDetection | None) -> bool | None:
    if pose is None or pose.court_position_m is None:
        return None
    return pose.court_position_m[1] < COURT_LENGTH_M / 2


def _pose_of(
    players: dict[int, list[PoseDetection]], frame: int, player_id: int
) -> PoseDetection | None:
    for offset in sorted(range(-MAX_POSE_GAP, MAX_POSE_GAP + 1), key=abs):
        for pose in players.get(frame + offset, []):
            if pose.player_id == player_id:
                return pose
    return None


def score_clip(
    clip: str,
    truth: list[HitMark],
    analysis: RallyAnalysis,
    frozen: set[int] | frozenset[int] = frozenset(),
) -> list[HitRow]:
    """Pair the system's hits with the labelled ones and judge each pair."""
    fps = analysis.fps
    shots = analysis.shots
    pairs = match_events(
        [s.frame_index / fps for s in shots], [m.frame / fps for m in truth], COLLAR_S
    )
    shot_of_truth = {t: p for p, t in pairs}
    paired_shots = {p for p, _ in pairs}

    rows: list[HitRow] = []
    for t, mark in enumerate(truth):
        in_freeze = mark.frame in frozen
        truth_type = mark.shot_type
        if t not in shot_of_truth:
            rows.append(
                HitRow(
                    clip,
                    mark.frame,
                    truth_type,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    in_freeze,
                )
            )
            continue
        shot = shots[shot_of_truth[t]]
        true_pose = clicked_player(mark, analysis.players)
        true_player = true_pose.player_id if true_pose is not None else None
        player_ok = shot.player_id is not None and shot.player_id == true_player
        team_ok: bool | None = None
        if shot.player_id is not None:
            system_side = _side_of_net(_pose_of(analysis.players, shot.frame_index, shot.player_id))
            true_side = _side_of_net(true_pose)
            if system_side is not None and true_side is not None:
                team_ok = system_side == true_side
        else:
            team_ok = False
        type_ok = None if truth_type not in CLASSES else shot.shot_type == truth_type
        rows.append(
            HitRow(
                clip,
                mark.frame,
                truth_type,
                shot.frame_index,
                shot.shot_type,
                true_player,
                shot.player_id,
                player_ok,
                team_ok,
                type_ok,
                in_freeze,
            )
        )
    for p, shot in enumerate(shots):
        if p not in paired_shots:
            rows.append(
                HitRow(
                    clip,
                    None,
                    None,
                    shot.frame_index,
                    shot.shot_type,
                    None,
                    shot.player_id,
                    None,
                    None,
                    None,
                    shot.frame_index in frozen,
                )
            )
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(rows: list[HitRow]) -> dict[str, float]:
    """Headline numbers for one set of rows (a clip, a group, or everything)."""
    labelled = [r for r in rows if r.truth_frame is not None]
    detected = [r for r in rows if r.system_frame is not None]
    paired = [r for r in labelled if r.system_frame is not None]
    tp = len(paired)
    precision = _ratio(tp, len(detected))
    recall = _ratio(tp, len(labelled))
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    typed = [r for r in paired if r.type_ok is not None]
    typed_right_player = [r for r in typed if r.player_ok]
    four_class = [r for r in labelled if r.truth_type in CLASSES]
    return {
        "labelled_hits": len(labelled),
        "detected_hits": len(detected),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "player_accuracy": _ratio(sum(1 for r in paired if r.player_ok), tp),
        "team_accuracy": _ratio(sum(1 for r in paired if r.team_ok), tp),
        "hitter_not_detected": sum(1 for r in paired if r.true_player is None),
        "type_accuracy": _ratio(sum(1 for r in typed if r.type_ok), len(typed)),
        "type_accuracy_right_player": _ratio(
            sum(1 for r in typed_right_player if r.type_ok), len(typed_right_player)
        ),
        "end_to_end": _ratio(
            sum(1 for r in four_class if r.player_ok and r.type_ok), len(four_class)
        ),
        "in_freeze": sum(1 for r in labelled if r.in_freeze),
    }


def type_confusion(rows: list[HitRow]) -> tuple[list[str], list[str | None], list[str]]:
    """(truth, prediction, labels) over paired hits with a four-class label."""
    labels = [*CLASSES, "Unclassified"]
    typed = [r for r in rows if r.type_ok is not None]
    return [r.truth_type or "" for r in typed], [r.system_type for r in typed], labels


def other_hits(rows: list[HitRow]) -> dict[str, int]:
    """What the system called the hits labelled "Other" (outside the taxonomy)."""
    return dict(
        collections.Counter(
            r.system_type or "Unclassified"
            for r in rows
            if r.truth_type == "Other" and r.system_frame is not None
        )
    )


def write_rows(rows: list[HitRow], path: Path) -> None:
    """Every hit, one row: the table behind every number, for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(HitRow.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

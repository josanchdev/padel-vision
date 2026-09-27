"""Scoring the whole chain against CVSPORTS ground truth (scripts/evaluate_chain_cv.py).

One `Row` per real hit and per unpaired detection, holding what the chain chose
under each audio-sync mode, so every metric of every mode is computed from the
same pairing. The definitions live here, not in the scripts, so the run and the
evidence that reports it cannot disagree on what a number means.
"""

from __future__ import annotations

from dataclasses import dataclass

from padel_ml.shot_type_dataset import CLASSES

MODES = ("none", "global", "video")


@dataclass
class Row:
    """One real hit or one detection, and how the chain did on it, per sync mode."""

    rally: str
    tournament: str
    truth_s: float | None
    """Annotated instant (window centre); None for a detection with no real hit."""
    detected_s: float | None
    """Raw audio instant; None for a missed hit."""
    truth_type: str | None
    truth_player: int | None
    """VIGO only."""
    player: dict[str, int | None]
    """Chosen player per sync mode."""
    shot_type: dict[str, str | None]


def ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def same_team(chosen: int | None, truth: int | None) -> bool:
    """Players 1-2 are one team and 3-4 the other (the paper's numbering)."""
    return chosen is not None and truth is not None and (chosen <= 2) == (truth <= 2)


def summarize(rows: list[Row], mode: str) -> dict[str, float]:
    real = [r for r in rows if r.truth_s is not None]
    detections = [r for r in rows if r.detected_s is not None]
    paired = [r for r in real if r.detected_s is not None]
    precision, recall = ratio(len(paired), len(detections)), ratio(len(paired), len(real))
    typed = [r for r in paired if r.truth_type in CLASSES]
    who = [r for r in paired if r.truth_player is not None]
    four_class = [r for r in real if r.truth_type in CLASSES]
    vigo = [r for r in four_class if r.truth_player is not None]
    right_player = [r for r in who if r.player[mode] == r.truth_player]
    return {
        "real_hits": len(real),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall
        else 0.0,
        "type_accuracy": ratio(sum(r.shot_type[mode] == r.truth_type for r in typed), len(typed)),
        "player_accuracy": ratio(len(right_player), len(who)),
        "team_accuracy": ratio(
            sum(same_team(r.player[mode], r.truth_player) for r in who), len(who)
        ),
        "type_accuracy_right_player": ratio(
            sum(r.shot_type[mode] == r.truth_type for r in right_player if r.truth_type in CLASSES),
            sum(1 for r in right_player if r.truth_type in CLASSES),
        ),
        "end_to_end_when_what": ratio(
            sum(r.detected_s is not None and r.shot_type[mode] == r.truth_type for r in four_class),
            len(four_class),
        ),
        "end_to_end_vigo": ratio(
            sum(
                r.detected_s is not None
                and r.player[mode] == r.truth_player
                and r.shot_type[mode] == r.truth_type
                for r in vigo
            ),
            len(vigo),
        ),
        "vigo_hits": len(vigo),
    }

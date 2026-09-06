"""Validate the audio hit-detector against Jorge's labeled shots (citys_cup).

Proof of concept: does the ball-hit show up as a clean audio peak, and do those
peaks line up with the 413 hand-marked shots? Reports recall/precision matching
each audio peak to a real shot within a tolerance, and how bounces/wall hits
compare. Run this once the audio file exists at data/raw/youtube/citys_cup_audio.*
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from padel_ml.audio_hits import AudioHit, detect_hits_in_audio
from padel_ml.shot_eval import load_shot_blocks


@dataclass
class AudioEval:
    n_real: int
    n_peaks: int
    tp: int
    recall: float
    precision: float


def evaluate_audio_hits(
    audio_path: Path,
    shots_csv: Path,
    fps: float,
    tol_s: float = 0.10,
    threshold: float = 0.3,
) -> AudioEval:
    """Match audio peaks to real shots (frames -> seconds via `fps`).

    A real shot at frame f is at time f/fps. An audio peak counts as a hit if it
    lands within `tol_s` of a real shot. `fps` matters: citys_cup is 60fps.
    """
    peaks = detect_hits_in_audio(audio_path, threshold=threshold)
    blocks = load_shot_blocks(shots_csv)
    real_times = sorted(b.centre / fps for b in blocks)

    matched: set[int] = set()
    tp = 0
    for pk in peaks:
        # nearest unmatched real shot within tolerance
        best_i, best_d = -1, tol_s
        for i, rt in enumerate(real_times):
            if i in matched:
                continue
            d = abs(rt - pk.time_s)
            if d <= best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            matched.add(best_i)
            tp += 1
    recall = len(matched) / len(real_times) if real_times else 0.0
    precision = tp / len(peaks) if peaks else 0.0
    return AudioEval(
        n_real=len(real_times),
        n_peaks=len(peaks),
        tp=tp,
        recall=recall,
        precision=precision,
    )


def peaks_near_real(
    peaks: list[AudioHit], real_times: list[float], tol_s: float
) -> tuple[int, int]:
    """(peaks matching a real shot, peaks not matching) — for a quick FP look."""
    near = sum(1 for p in peaks if any(abs(p.time_s - rt) <= tol_s for rt in real_times))
    return near, len(peaks) - near

"""TrackNet detection metrics: peak distance with a tolerance radius.

A ball prediction is not scored per pixel but by where its heatmap PEAK lands
relative to the true ball (the TrackNet convention). Each frame is one of:

- TP: ball present, predicted peak within `tol` cells of the true centre
- FP1: ball present but predicted peak too far (localized wrong)
- FP2: ball absent but the model fired a peak anyway (hallucination)
- FN: ball present but no peak above threshold (missed)
- TN: ball absent and no peak (correct silence)

From these come precision/recall/F1 and the mean localization error (in grid
cells) over the true positives. We also split the frames flagged `occluded` in
PadelTracker100 from the visible ones, because the hypothesis (ADR-0009) is that
TrackNetV3 recovers occluded balls that V2 misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float32]


def peak_xy(heatmap: torch.Tensor) -> tuple[int, int]:
    """(H, W) heatmap -> (x, y) of its argmax cell."""
    flat = int(heatmap.argmax().item())
    w = heatmap.shape[-1]
    return flat % w, flat // w


def is_present(heatmap: torch.Tensor) -> bool:
    """A ground-truth heatmap encodes a ball iff it has any mass."""
    return bool(heatmap.max() > 0)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    errors: list[float] = field(default_factory=list)  # localization error, TP only

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0


@dataclass
class BallEval:
    overall: Counts
    visible: Counts
    occluded: Counts


def _score_frame(
    counts: Counts, pred: torch.Tensor, true: torch.Tensor, threshold: float, tol: float
) -> None:
    present = is_present(true)
    fired = bool(pred.max() > threshold)
    if not present:
        if fired:
            counts.fp += 1  # hallucinated a ball where there is none
        else:
            counts.tn += 1
        return
    if not fired:
        counts.fn += 1  # missed a real ball
        return
    px, py = peak_xy(pred)
    tx, ty = peak_xy(true)
    dist = float(np.hypot(px - tx, py - ty))
    if dist <= tol:
        counts.tp += 1
        counts.errors.append(dist)
    else:
        counts.fp += 1  # detected, but localized wrong


def evaluate_ball(
    preds: torch.Tensor,
    trues: torch.Tensor,
    occluded: npt.NDArray[np.bool_],
    threshold: float = 0.5,
    tol: float = 4.0,
) -> BallEval:
    """Score a batch of (N, H, W) predicted/true heatmaps.

    occluded[i] flags frames the dataset marked as ball-occluded, so the
    occluded-vs-visible breakdown can be reported. tol is in grid cells.
    """
    overall, visible, occ = Counts(), Counts(), Counts()
    for i in range(preds.shape[0]):
        pred, true = preds[i], trues[i]
        _score_frame(overall, pred, true, threshold, tol)
        _score_frame(occ if occluded[i] else visible, pred, true, threshold, tol)
    return BallEval(overall=overall, visible=visible, occluded=occ)

import numpy as np
import torch
from padel_ml.ball_metrics import evaluate_ball, is_present, peak_xy


def _blob(h, w, x, y, val=1.0) -> torch.Tensor:
    hm = torch.zeros(h, w)
    hm[y, x] = val
    return hm


def test_peak_xy_and_present() -> None:
    hm = _blob(10, 20, x=7, y=3)
    assert peak_xy(hm) == (7, 3)
    assert is_present(hm)
    assert not is_present(torch.zeros(10, 20))


def test_true_positive_within_tolerance() -> None:
    pred = _blob(10, 20, x=8, y=3)  # 1 cell off
    true = _blob(10, 20, x=7, y=3)
    res = evaluate_ball(pred[None], true[None], np.array([False]), tol=4.0)
    assert res.overall.tp == 1
    assert res.overall.f1 == 1.0
    assert res.overall.mean_error == 1.0


def test_mislocalized_is_false_positive() -> None:
    pred = _blob(10, 20, x=18, y=9)  # far from truth
    true = _blob(10, 20, x=1, y=1)
    res = evaluate_ball(pred[None], true[None], np.array([False]), tol=4.0)
    assert res.overall.tp == 0
    assert res.overall.fp == 1


def test_hallucination_when_ball_absent() -> None:
    pred = _blob(10, 20, x=5, y=5, val=0.9)
    true = torch.zeros(10, 20)  # no ball
    res = evaluate_ball(pred[None], true[None], np.array([False]), threshold=0.5)
    assert res.overall.fp == 1
    assert res.overall.tn == 0


def test_missed_ball_is_false_negative() -> None:
    pred = torch.zeros(10, 20)  # nothing fired
    true = _blob(10, 20, x=5, y=5)
    res = evaluate_ball(pred[None], true[None], np.array([False]), threshold=0.5)
    assert res.overall.fn == 1
    assert res.overall.recall == 0.0


def test_true_negative_when_both_empty() -> None:
    res = evaluate_ball(
        torch.zeros(1, 10, 20), torch.zeros(1, 10, 20), np.array([False]), threshold=0.5
    )
    assert res.overall.tn == 1


def test_occluded_visible_breakdown() -> None:
    # Frame 0 visible TP, frame 1 occluded FN.
    preds = torch.stack([_blob(10, 20, 5, 5), torch.zeros(10, 20)])
    trues = torch.stack([_blob(10, 20, 5, 5), _blob(10, 20, 8, 8)])
    res = evaluate_ball(preds, trues, np.array([False, True]), threshold=0.5)
    assert res.visible.tp == 1 and res.visible.fn == 0
    assert res.occluded.tp == 0 and res.occluded.fn == 1

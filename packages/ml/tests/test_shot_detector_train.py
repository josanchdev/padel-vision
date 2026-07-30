import numpy as np
from padel_ml.shot_detector_train import _metrics


def test_metrics_perfect() -> None:
    probs = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    ev = _metrics(probs, y, 0.5)
    assert ev.precision == 1.0
    assert ev.recall == 1.0
    assert ev.f1 == 1.0
    assert ev.accuracy == 1.0


def test_metrics_all_missed() -> None:
    probs = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    y = np.array([1, 1, 1], dtype=np.int64)
    ev = _metrics(probs, y, 0.5)
    assert ev.recall == 0.0
    assert ev.f1 == 0.0


def test_metrics_threshold_shifts_recall() -> None:
    probs = np.array([0.4, 0.6, 0.4, 0.6], dtype=np.float32)
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    lax = _metrics(probs, y, 0.3)  # everything predicted positive
    strict = _metrics(probs, y, 0.5)
    assert lax.recall == 1.0
    assert strict.recall == 0.5  # only the 0.6 positive survives

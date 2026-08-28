from padel_ml.shot_localizer_train import ShotEventLoc, peaks_from_probs


def test_peaks_one_per_shot() -> None:
    # Two clear peaks separated by a valley -> two events.
    probs = {f: 0.1 for f in range(100)}
    for f in range(18, 23):
        probs[f] = 0.9 if f == 20 else 0.6
    for f in range(58, 63):
        probs[f] = 0.95 if f == 60 else 0.6
    ev = peaks_from_probs(probs, threshold=0.5, min_gap=8)
    assert [e.frame for e in ev] == [20, 60]


def test_peaks_merge_close_maxima() -> None:
    # Two maxima 3 frames apart -> collapse to the stronger (min_gap=8).
    probs = {f: 0.1 for f in range(50)}
    probs[20] = 0.7
    probs[23] = 0.9
    ev = peaks_from_probs(probs, threshold=0.5, min_gap=8)
    assert len(ev) == 1
    assert ev[0] == ShotEventLoc(23, 0.9)


def test_peaks_none_below_threshold() -> None:
    probs = {f: 0.3 for f in range(30)}
    assert peaks_from_probs(probs, threshold=0.5) == []


def test_averaged_probs_smooths_via_overlap(monkeypatch) -> None:
    """sliding_probs_averaged should average the overlapping window votes per
    frame. With a model that always predicts 0.5, every frame ends up 0.5."""
    import numpy as np
    from padel_ml import shot_localizer_train as slt

    class _ConstModel:
        def to(self, _d):
            return self

        def eval(self):
            return self

        def __call__(self, pose, ball):
            import torch

            t = pose.shape[1]
            return torch.zeros(1, t)  # sigmoid(0)=0.5

    # a window is always available
    def _fake_window(c, p, b):
        return np.zeros((32, 17, 3), np.float32), np.zeros((32, 3), np.float32)

    monkeypatch.setattr(slt, "_window_around", _fake_window)
    out = slt.sliding_probs_averaged(_ConstModel(), {}, {}, 0, 40, step=4, device="cpu")
    assert out
    assert all(abs(v - 0.5) < 1e-5 for v in out.values())


def test_min_run_drops_isolated_spikes() -> None:
    # An isolated 1-frame spike vs a sustained run. min_run=4 keeps only the run.
    probs = {f: 0.1 for f in range(100)}
    probs[20] = 0.9  # lone spike
    for f in range(50, 56):  # sustained run around 52
        probs[f] = 0.9
    lenient = peaks_from_probs(probs, threshold=0.5, min_gap=8, min_run=1)
    strict = peaks_from_probs(probs, threshold=0.5, min_gap=8, min_run=4, run_window=5)
    assert 20 in [e.frame for e in lenient]  # spike survives min_run=1
    assert 20 not in [e.frame for e in strict]  # spike dropped by min_run=4
    assert any(50 <= e.frame <= 55 for e in strict)  # the run survives

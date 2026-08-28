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

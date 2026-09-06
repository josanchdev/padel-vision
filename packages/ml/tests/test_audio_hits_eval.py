import csv

from padel_ml.audio_hits import AudioHit
from padel_ml.audio_hits_eval import peaks_near_real


def test_peaks_near_real_splits_matched_and_not() -> None:
    real = [1.0, 2.0, 3.0]  # seconds
    peaks = [AudioHit(1.02, 0.9), AudioHit(2.5, 0.8), AudioHit(3.01, 0.7)]
    near, far = peaks_near_real(peaks, real, tol_s=0.1)
    assert near == 2  # 1.02 and 3.01 match; 2.5 doesn't
    assert far == 1


def test_evaluate_audio_hits_matches_by_time(tmp_path, monkeypatch) -> None:
    # a shots CSV (quick-mark format) at 60fps: frames 60 and 120 -> 1.0s, 2.0s
    csv_path = tmp_path / "shots.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["frame", "type", "from_wall", "player"])
        w.writerow([60, "Forehand", 0, -1])
        w.writerow([120, "Backhand", 0, -1])

    from padel_ml import audio_hits_eval as ae

    # stub the detector to return peaks near the real times + one false peak
    monkeypatch.setattr(
        ae,
        "detect_hits_in_audio",
        lambda *a, **k: [AudioHit(1.01, 0.9), AudioHit(2.02, 0.8), AudioHit(5.0, 0.6)],
    )
    res = ae.evaluate_audio_hits(tmp_path / "x.m4a", csv_path, fps=60.0, tol_s=0.1)
    assert res.n_real == 2
    assert res.tp == 2  # both real shots hit
    assert res.recall == 1.0
    assert res.n_peaks == 3
    assert abs(res.precision - 2 / 3) < 1e-6  # one false peak

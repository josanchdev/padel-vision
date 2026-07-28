import json

from padel_ml.ball_generalization import CourtResult, plot_rates


def test_court_result_fields() -> None:
    r = CourtResult(court="wpt", frames=100, detections=80, detection_rate=0.8, mean_confidence=0.9)
    assert r.detection_rate == 0.8
    assert r.mean_confidence == 0.9


def test_plot_rates_writes_png(tmp_path) -> None:
    results = [
        CourtResult("wpt", 100, 90, 0.90, 0.95),
        CourtResult("black_court", 100, 30, 0.30, 0.55),
    ]
    out = tmp_path / "gen.png"
    plot_rates(results, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_json_roundtrip(tmp_path) -> None:
    from dataclasses import asdict

    results = [CourtResult("wpt", 100, 90, 0.90, 0.95)]
    path = tmp_path / "gen.json"
    path.write_text(json.dumps([asdict(r) for r in results]))
    loaded = json.loads(path.read_text())
    assert loaded[0]["court"] == "wpt"
    assert loaded[0]["detection_rate"] == 0.90

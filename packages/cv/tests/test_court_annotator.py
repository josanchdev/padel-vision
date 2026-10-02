from pathlib import Path

import numpy as np
import pytest

from padel_cv import court_annotator
from padel_cv.court_annotator import court_or_mark


@pytest.fixture
def marked(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the 6-click window with one that saves at once; record each call."""
    calls: list[Path] = []

    def fake_annotate(video_path: Path, out_json: Path, frame_index: int = 30) -> np.ndarray:
        calls.append(out_json)
        out_json.write_text("{}")
        return np.eye(3)

    monkeypatch.setattr(court_annotator, "annotate_court", fake_annotate)
    return calls


def test_a_marked_tournament_opens_no_window(tmp_path: Path, marked: list[Path]) -> None:
    (tmp_path / "20230528_VIGO.json").write_text("{}")
    court = court_or_mark(Path("20230528_VIGO_03.mp4"), tmp_path)
    assert court == tmp_path / "20230528_VIGO.json"
    assert marked == []


def test_an_unmarked_rally_is_saved_as_its_tournament(tmp_path: Path, marked: list[Path]) -> None:
    """Marking one rally serves the whole tournament: the camera is fixed."""
    court = court_or_mark(Path("20230528_VIGO_03.mp4"), tmp_path)
    assert court == tmp_path / "20230528_VIGO.json"
    assert court_or_mark(Path("20230528_VIGO_07.mp4"), tmp_path) == court
    assert len(marked) == 1


def test_a_clip_outside_a_tournament_gets_its_own_file(tmp_path: Path, marked: list[Path]) -> None:
    assert court_or_mark(Path("final_menorca.mp4"), tmp_path) == tmp_path / "final_menorca.json"


def test_cancelling_the_marking_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(court_annotator, "annotate_court", lambda *args, **kwargs: None)
    assert court_or_mark(Path("20230528_VIGO_03.mp4"), tmp_path) is None

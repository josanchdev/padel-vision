from padel_cv.court_registry import court_file_for, tournament_of


def test_tournament_from_rally_name() -> None:
    assert tournament_of("20230528_VIGO_03") == "20230528_VIGO"
    assert tournament_of("20231015_AMSTERDAM_00") == "20231015_AMSTERDAM"


def test_uses_the_tournament_court(tmp_path) -> None:
    (tmp_path / "20230528_VIGO.json").write_text("{}")
    assert court_file_for("20230528_VIGO_03", tmp_path).name == "20230528_VIGO.json"


def test_range_file_wins_from_its_rally_on(tmp_path) -> None:
    """VALLADOLID's camera rises at rally 04; later rallies need the second mark."""
    (tmp_path / "20230702_VALLADOLID.json").write_text("{}")
    (tmp_path / "20230702_VALLADOLID@04.json").write_text("{}")
    assert court_file_for("20230702_VALLADOLID_03", tmp_path).name == "20230702_VALLADOLID.json"
    assert court_file_for("20230702_VALLADOLID_04", tmp_path).name == "20230702_VALLADOLID@04.json"
    assert court_file_for("20230702_VALLADOLID_05", tmp_path).name == "20230702_VALLADOLID@04.json"


def test_unmarked_tournament_returns_none(tmp_path) -> None:
    assert court_file_for("20230528_VIGO_00", tmp_path) is None

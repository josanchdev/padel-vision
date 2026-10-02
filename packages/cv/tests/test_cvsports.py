from pathlib import Path

from padel_cv.cvsports import load_hits_csv, parse_rally, tournament_of


def test_tournament_from_rally_name() -> None:
    assert tournament_of("20230528_VIGO_03") == "20230528_VIGO"
    assert tournament_of("20231015_AMSTERDAM_00") == "20231015_AMSTERDAM"


def test_a_clip_outside_the_dataset_is_its_own_tournament() -> None:
    assert tournament_of("final_menorca") == "final_menorca"
    assert parse_rally("final_menorca") is None


def test_rally_index_is_parsed() -> None:
    assert parse_rally("20230702_VALLADOLID_04") == ("20230702_VALLADOLID", 4)


def test_hits_are_grouped_by_rally(tmp_path: Path) -> None:
    csv_path = tmp_path / "hits.csv"
    csv_path.write_text(
        "filename,start,end,class,class_id\n"
        "a.mp4,1.0,1.2,hit,0\n"
        "b.mp4,2.0,2.5,hit,0\n"
        "a.mp4,3.0,3.1,hit,0\n"
    )
    assert load_hits_csv(csv_path) == {"a.mp4": [(1.0, 1.2), (3.0, 3.1)], "b.mp4": [(2.0, 2.5)]}

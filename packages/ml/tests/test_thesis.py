import json
import shutil
from pathlib import Path

import pytest
from padel_ml.thesis import tables
from padel_ml.thesis.figures import FIGURES, STYLE
from padel_ml.thesis.latex import (
    Number,
    fixed,
    numbers_file,
    percent,
    percent_range,
    proportion,
    tabular,
)
from padel_ml.thesis.numbers import thesis_numbers
from padel_ml.thesis.sources import Evidence

from padel_cv.paths import REPO


def test_rounding_is_half_up_as_by_hand() -> None:
    # round() would give 83.4 (0.8345 * 100 is 83.44999... in floating point) and 8.2.
    assert fixed(0.8345, 1, scale=100) == "83.5"
    assert fixed(0.0825, 1, scale=100) == "8.3"
    assert fixed(0.95816, 3) == "0.958"


def test_negative_numbers_keep_their_sign_but_never_print_minus_zero() -> None:
    assert fixed(-0.054, 1, scale=100) == "-5.4"
    assert fixed(-0.0004, 1, scale=100) == "0.0"


def test_percentages_take_one_decimal_and_proportions_three() -> None:
    assert percent(0.8271) == r"\SI{82.7}{\percent}"
    assert proportion(0.9582) == r"\num{0.958}"
    assert percent_range(0.8033, 0.8497) == r"\num{80.3}--\num{85.0}"


def test_numbers_file_defines_every_key_once() -> None:
    text = numbers_file([Number("tipo-accuracy", "x", "a.json: b"), Number("audio-f1", "y", "c")])
    assert r"\newcommand{\cifra}" in text
    assert r"\@namedef{cifra@tipo-accuracy}{x}" in text


@pytest.mark.parametrize(
    "key", ["TipoAccuracy", "tipo_accuracy", "tipo-", "-tipo", "tipo accuracy"]
)
def test_numbers_file_rejects_keys_latex_or_the_reader_would_trip_on(key: str) -> None:
    with pytest.raises(ValueError, match="bad key"):
        numbers_file([Number(key, "x", "source")])


def test_numbers_file_rejects_a_key_defined_twice() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        numbers_file([Number("audio-f1", "x", "a"), Number("audio-f1", "y", "b")])


def test_tabular_has_booktabs_rules_and_no_vertical_ones() -> None:
    text = tabular("lc", ["Medida", "Valor"], [["a", "1"], ["b", "2"]], rules=[1])
    assert text.count(r"\midrule") == 2
    assert r"\toprule" in text and r"\bottomrule" in text
    assert "|" not in text


def test_tabular_rejects_a_row_with_the_wrong_number_of_cells() -> None:
    with pytest.raises(ValueError, match="3 cells"):
        tabular("lc", ["Medida", "Valor"], [["a", "1", "extra"]])


def test_every_number_of_the_text_reads_from_the_real_evidence() -> None:
    numbers = thesis_numbers(Evidence())
    text = numbers_file(numbers)  # raises on a bad or repeated key
    assert len(numbers) == text.count(r"\@namedef{cifra@")
    assert all(n.source and "json" in n.source for n in numbers)


def test_evidence_reads_the_three_bounds_of_an_interval(tmp_path: Path) -> None:
    metrics = {"t.m.valor": 0.5, "t.m.bajo": 0.4, "t.m.alto": 0.6}
    (tmp_path / "confidence_intervals.json").write_text(json.dumps({"metrics": metrics}))
    assert Evidence(tmp_path).interval("t.m") == (0.5, 0.4, 0.6)


@pytest.mark.parametrize(
    "build",
    [
        tables.steps_table,
        tables.system_table,
        tables.decisions_table,
        tables.ball_table,
        tables.cost_table,
        tables.datasets_table,
    ],
)
def test_every_table_builds_from_the_real_evidence(build: object) -> None:
    text = build(Evidence())  # type: ignore[operator]
    assert text.startswith(r"\begin{tabular}") and text.endswith(r"\end{tabular}")


def test_tools_table_reads_versions_from_the_lock_files() -> None:
    text = tables.tools_table(REPO / "uv.lock", REPO / "packages" / "web" / "package-lock.json")
    assert "PyTorch" in text and "React" in text


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="figures are set by pdflatex")
def test_a_figure_is_written_as_pdf(tmp_path: Path) -> None:
    import matplotlib as mpl

    out = tmp_path / "embudo.pdf"
    with mpl.rc_context(STYLE):
        FIGURES["embudo.pdf"](Evidence(), out)
    assert out.read_bytes().startswith(b"%PDF")

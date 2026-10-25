import pytest
from padel_ml.thesis.latex import fixed, percent, percent_range, proportion, tabular


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


def test_tabular_has_booktabs_rules_and_no_vertical_ones() -> None:
    text = tabular("lc", ["Medida", "Valor"], [["a", "1"], ["b", "2"]], rules=[1])
    assert text.count(r"\midrule") == 2
    assert r"\toprule" in text and r"\bottomrule" in text
    assert "|" not in text


def test_tabular_turns_a_plain_string_into_a_heading_across_the_table() -> None:
    text = tabular("lc", ["Medida", "Valor"], ["Los 11 torneos", ["a", "1"]])
    assert r"\multicolumn{2}{@{}l}{\emph{Los 11 torneos}} \\" in text
    assert r"\textbf{Medida} & \textbf{Valor}" in text


def test_tabular_rejects_a_row_with_the_wrong_number_of_cells() -> None:
    with pytest.raises(ValueError, match="3 cells"):
        tabular("lc", ["Medida", "Valor"], [["a", "1", "extra"]])

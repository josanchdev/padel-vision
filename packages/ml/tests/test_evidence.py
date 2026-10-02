import numpy as np
from padel_ml.evidence import class_scores, macro_f1, per_class_metrics


def test_class_scores_come_from_rows_of_truth_and_columns_of_prediction() -> None:
    matrix = np.array([[2, 1], [0, 3]])
    scores = class_scores(matrix, ["a", "b"])
    assert scores["a"]["recall"] == 2 / 3
    assert scores["b"]["precision"] == 3 / 4
    assert scores["a"]["support"] == 3


def test_the_report_rounds_what_class_scores_computes() -> None:
    report = per_class_metrics(np.array([[2, 1], [0, 3]]), ["a", "b"])
    assert report["a"]["recall"] == 0.6667
    assert report["b"]["support"] == 3
    assert macro_f1(report) == round((0.8 + 0.8571) / 2, 4)

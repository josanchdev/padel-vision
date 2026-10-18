"""Write the per-class table of section 5.1.4: memoria/figuras/tabla-tipo-clases.tex.

Precision, recall, F1 and number of hits of each shot type, with the serve rule,
over the 2186 hits classified one tournament out at a time; the last row is the
macro mean. From shot_type_classifier.json (scripts/evidence_shot_classifier.py).

    uv run python scripts/build_shot_type_classes_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.shot_type_dataset import CLASSES
from padel_ml.thesis.latex import count, proportion, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-tipo-clases.tex"


def build_table() -> str:
    evidence = json.loads((METRICS / "shot_type_classifier.json").read_text())
    per_class = evidence["per_class"]
    rows: list[list[str] | str] = [
        [
            name,
            proportion(per_class[key]["precision"]),
            proportion(per_class[key]["recall"]),
            proportion(per_class[key]["f1"]),
            count(per_class[key]["support"]),
        ]
        for key, name in zip(CLASSES, evidence["labels"], strict=True)
    ]
    metrics = evidence["metrics"]
    rows.append(["Media (macro)", "", "", proportion(metrics["macro_f1"]), count(metrics["n"])])
    header = ["Clase", "Precisión", r"\textit{Recall}", "F1", "Golpes"]
    return tabular("lcccc", header, rows, rules=[len(CLASSES)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

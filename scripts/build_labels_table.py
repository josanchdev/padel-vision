"""Write the labelling table of section 3.2: memoria/figuras/tabla-etiquetado.tex.

How many of the 2377 CVSPORTS hits got each type in our labelling, and their
share. "Otro" is the discard label (`Other`): hits that are none of the four
classes. From dataset_shot_types.json (scripts/evidence_dataset_stats.py).

    uv run python scripts/build_labels_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import count, percent, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-etiquetado.tex"
CLASSES = [
    ("Forehand", "Derecha"),
    ("Backhand", "Revés"),
    ("Smash", "Remate"),
    ("Serve", "Saque"),
    ("Other", "Otro"),
]


def build_table() -> str:
    evidence = json.loads((METRICS / "dataset_shot_types.json").read_text())
    total = evidence["metrics"]["labelled"]
    support = {key: values["support"] for key, values in evidence["per_class"].items()}
    if sum(support.values()) != total:
        raise SystemExit(f"the classes add up to {sum(support.values())}, not {total}")
    rows: list[list[str] | str] = [
        [name, count(support[key]), percent(support[key] / total)] for key, name in CLASSES
    ]
    rows.append(["Total", count(total), ""])
    return tabular("lcc", ["Clase", "Golpes", "Porcentaje"], rows, rules=[len(CLASSES)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

"""Write the dataset table of section 3.1: memoria/figuras/tabla-dataset.tex.

Rallies and hits per tournament, counted from the dataset's own files
(`metadata/hits.csv` and the rally videos). The setting (indoor or outdoor) is
not in those files: it is the one the dataset's paper gives for each tournament.

    uv run python scripts/build_dataset_table.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from padel_ml.thesis.figures import TOURNAMENT_NAMES
from padel_ml.thesis.latex import tabular

from padel_cv.cvsports import tournament_of
from padel_cv.paths import CVSPORTS_HITS, CVSPORTS_RALLIES, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-dataset.tex"

OUTDOOR = {"VIGO", "BRUSSEL", "VALLADOLID"}
"""Tournaments the dataset's paper lists as played outdoors; the rest are indoors."""


def count_per_tournament() -> tuple[Counter[str], Counter[str]]:
    """(rallies, hits) per tournament, keyed by its name without the date."""
    rallies: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for video in CVSPORTS_RALLIES.glob("*.mp4"):
        rallies[tournament_of(video.stem).split("_", 1)[1]] += 1
    with CVSPORTS_HITS.open() as handle:
        for row in csv.DictReader(handle):
            hits[tournament_of(Path(row["filename"]).stem).split("_", 1)[1]] += 1
    return rallies, hits


def build_table() -> str:
    rallies, hits = count_per_tournament()
    order = sorted(hits, key=lambda name: -hits[name])
    rows: list[list[str] | str] = [
        [
            TOURNAMENT_NAMES[name],
            str(rallies[name]),
            str(hits[name]),
            "Exterior" if name in OUTDOOR else "Interior",
        ]
        for name in order
    ]
    rows.append(
        [
            r"\textbf{Total}",
            rf"\textbf{{{sum(rallies.values())}}}",
            rf"\textbf{{{sum(hits.values())}}}",
            "",
        ]
    )
    return tabular("lrrl", ["Torneo", "Puntos", "Golpes", "Entorno"], rows, rules=[len(rows) - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

"""Write the hit detection table of section 4.2.1: memoria/figuras/tabla-deteccion.tex.

Image against sound on the same five points, from detection_comparison.json
(scripts/evidence_detection_comparison.py): precision, recall and F1 with the
same 250 ms rule, and seconds of processing per minute of video.

    uv run python scripts/build_detection_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import number, proportion, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-deteccion.tex"
SECONDS_PLACES = {"image": 1, "audio": 2}
"""The audio takes a fraction of a second per minute; one decimal would print 0,0."""


def build_table() -> str:
    metrics = json.loads((METRICS / "detection_comparison.json").read_text())["metrics"]
    rows: list[list[str] | str] = [
        [
            name,
            proportion(metrics[f"{key}_precision"]),
            proportion(metrics[f"{key}_recall"]),
            proportion(metrics[f"{key}_f1"]),
            number(metrics[f"{key}_s_per_min"], SECONDS_PLACES[key]),
        ]
        for key, name in (("image", "Imagen (pose y pelota)"), ("audio", "Audio"))
    ]
    header = ["Entrada", "Precisión", r"\textit{Recall}", "F1", "Tiempo (s/min)"]
    return tabular("lcccc", header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

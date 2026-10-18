"""Write the hit detection table of section 4.2.1: memoria/figuras/tabla-deteccion.tex.

Image against sound on the same five points, from detection_comparison.json
(scripts/evidence_detection_comparison.py): precision, recall and F1 with the
same 250 ms rule. No time column: the pose and the ball are computed anyway for
the rest of the system, so the image is not a cheaper alternative; what the
sound adds is told in the text.

    uv run python scripts/build_detection_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import proportion, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-deteccion.tex"


def build_table() -> str:
    metrics = json.loads((METRICS / "detection_comparison.json").read_text())["metrics"]
    rows: list[list[str] | str] = [
        [
            name,
            proportion(metrics[f"{key}_precision"]),
            proportion(metrics[f"{key}_recall"]),
            proportion(metrics[f"{key}_f1"]),
        ]
        for key, name in (("image", "Imagen (pose y pelota)"), ("audio", "Audio"))
    ]
    header = ["Entrada", "Precisión", r"\textit{Recall}", "F1"]
    return tabular("lccc", header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

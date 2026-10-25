"""Write the table of section 4.4: memoria/figuras/tabla-uso-datos.tex.

Which data each component of the system uses, and for what. The content is
written here, in one place, so the table comes from code like the others; it
states how each component is trained or evaluated, not measured numbers.

    uv run python scripts/build_data_usage_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padel_ml.thesis.latex import tabular

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "tabla-uso-datos.tex"

ROWS: list[list[str]] = [
    [
        "Detector de golpes",
        "CVSPORTS (audio e instantes)",
        "Entrenamiento y evaluación",
    ],
    [
        "Detector de la pelota",
        "PadelTracker100 (dos finales)",
        "Entrenamiento y evaluación",
    ],
    [
        "Asignación del golpe",
        "CVSPORTS (Vigo)",
        "Evaluación",
    ],
    [
        "Clasificador del tipo",
        "Etiquetado propio",
        "Entrenamiento y evaluación",
    ],
]
SPEC = "p{3.8cm}p{5.4cm}p{4.8cm}"
HEADER = ["Componente", "Datos", "Uso"]


def build_table() -> str:
    return tabular(SPEC, HEADER, ROWS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

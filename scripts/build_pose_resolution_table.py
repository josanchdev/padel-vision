"""Write the pose input size table of section 4.2.2: memoria/figuras/tabla-pose-resolucion.tex.

From pose_resolution.json (scripts/evidence_pose_resolution.py): for each input
size the pose model can shrink the frame to, the share of frames with the four
players inside the court mask, and the time per frame.

    uv run python scripts/build_pose_resolution_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import number, percent, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-pose-resolucion.tex"


def build_table() -> str:
    evidence = json.loads((METRICS / "pose_resolution.json").read_text())
    metrics = evidence["metrics"]
    rows: list[list[str] | str] = [
        [
            rf"\num{{{size}}}~px",
            percent(metrics[f"all_four_{size}"]),
            number(metrics[f"ms_per_frame_{size}"], 1),
        ]
        for size in evidence["params"]["sizes"]
    ]
    header = ["Tamaño de entrada", "Cuatro jugadores", "Tiempo (ms/fotograma)"]
    return tabular("lcc", header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

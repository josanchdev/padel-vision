"""Write the pose input size table of section 4.2.2: memoria/figuras/tabla-pose-resolucion.tex.

From pose_resolution.json (scripts/evidence_pose_resolution.py): for each input
size the pose model can shrink the frame to, the share of frames with the four
players inside the court mask, and the extra time per frame over the smallest
size. The time is relative because its absolute value depends on the machine.

    uv run python scripts/build_pose_resolution_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import fixed, percent, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-pose-resolucion.tex"


def extra_time(ms: float, base_ms: float) -> str:
    """Time over the base size, as a signed percentage; the base itself is a dash."""
    if ms == base_ms:
        return "---"
    return rf"\SI[retain-explicit-plus]{{+{fixed(ms / base_ms - 1, 1, scale=100)}}}{{\percent}}"


def build_table() -> str:
    evidence = json.loads((METRICS / "pose_resolution.json").read_text())
    metrics = evidence["metrics"]
    sizes = evidence["params"]["sizes"]
    base_ms = metrics[f"ms_per_frame_{sizes[0]}"]
    rows: list[list[str] | str] = [
        [
            rf"\num{{{size}}}~px",
            percent(metrics[f"all_four_{size}"]),
            extra_time(metrics[f"ms_per_frame_{size}"], base_ms),
        ]
        for size in sizes
    ]
    header = ["Tamaño de entrada", "Cuatro jugadores", "Tiempo adicional"]
    return tabular("lcc", header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

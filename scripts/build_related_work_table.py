"""Write the table of section 2.1: memoria/figuras/tabla-trabajos-padel.tex.

What each published padel video system extracts. Not measured data: each cell
was read in the paper itself, and the source of every cell is noted below, so
the table can be checked line by line.

    uv run python scripts/build_related_work_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padel_ml.thesis.latex import tabular

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "tabla-trabajos-padel.tex"

AUTO = r"$\checkmark$"
MANUAL = "A mano"
NO = "---"


def stacked(top: str, bottom: str) -> str:
    """A header on two lines, so the seven columns fit the text width."""
    return rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}{top}\\{bottom}\end{{tabular}}"


COLUMNS = [
    "Pista",
    "Jugadores",
    "Pelota",
    stacked("Cuándo", "se golpea"),
    stacked("Quién", "golpea"),
    stacked("Tipo de", "golpe"),
]
WORKS: list[tuple[str, str, list[str]]] = [
    # Javadiha et al. (2021), Sensors 21(10):3368. Court: the four corners are taken
    # by hand for the perspective transform (§2.3, p. 5). Players: positions from
    # pose keypoints (abstract, table 3). No ball, hits or shot types.
    (r"Javadiha \textit{et al.}", "javadiha2021positions", [MANUAL, AUTO, NO, NO, NO, NO]),
    # Novillo et al. (2025), IDEAL 2024, LNCS 15346. Court detected automatically
    # by colour clustering and contours (§2.1); players with YOLOv8 (§2.2); ball
    # with TrackNet (§2.3). Shot types only as future work (§4, p. 119).
    (r"Novillo \textit{et al.}", "novillo2025padel", [AUTO, AUTO, AUTO, NO, NO, NO]),
    # Decorte et al. (2024), CVPRW. Court from manually selected points (§5.1);
    # players with YOLO pose (§5.2); ball with TrackNet trained on tennis (§4.3);
    # hits from audio (§4.2); hitter by a weighted vote (§5.3). No shot types.
    (r"Decorte \textit{et al.}", "decorte2024padel", [MANUAL, AUTO, AUTO, AUTO, AUTO, NO]),
]


def build_table() -> str:
    rows: list[list[str] | str] = [
        [rf"{name}~\cite{{{key}}}", *cells] for name, key, cells in WORKS
    ]
    return tabular("l" + "c" * len(COLUMNS), ["Trabajo", *COLUMNS], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

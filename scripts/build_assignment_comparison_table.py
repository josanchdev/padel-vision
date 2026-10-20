"""Write the hit assignment table of section 5.1.3: memoria/figuras/tabla-asignacion-paper.tex.

Player and team accuracy on the 319 hits of the paper's Table 3 (16 VIGO
rallies, annotated instants), next to Decorte et al. Ours and its 95 % interval
come from confidence_intervals.json (scripts/evidence_confidence_intervals.py);
the paper's figures from hit_assignment_replica.json
(scripts/evidence_hit_assignment.py). The difference is in percentage points.

    uv run python scripts/build_assignment_comparison_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import fixed, percent, percent_range, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-asignacion-paper.tex"


def signed_points(share: float) -> str:
    """A difference of proportions in percentage points, sign kept: 0.0376 -> +3,8."""
    text = fixed(share, 1, scale=100)
    sign = "" if text.startswith("-") else "+"
    return rf"\num[retain-explicit-plus]{{{sign}{text}}}"


def build_table() -> str:
    intervals = json.loads((METRICS / "confidence_intervals.json").read_text())["metrics"]
    paper = json.loads((METRICS / "hit_assignment_replica.json").read_text())["metrics"]
    rows: list[list[str] | str] = []
    for key, name in (("player", "Jugador"), ("team", "Pareja")):
        prefix = f"tabla1_asignacion.{key}"
        rows.append(
            [
                name,
                percent(intervals[f"{prefix}.valor"]),
                percent_range(intervals[f"{prefix}.bajo"], intervals[f"{prefix}.alto"]),
                percent(paper[f"paper_{key}_accuracy"]),
                signed_points(intervals[f"{prefix}_minus_paper.valor"]),
            ]
        )
    header = [
        "Acierto",
        "Este trabajo",
        r"Intervalo 95~\%",
        r"Decorte \textit{et al.}",
        "Diferencia",
    ]
    return tabular("lcccc", header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

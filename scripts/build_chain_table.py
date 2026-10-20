"""Write the whole chain table of section 5.2: memoria/figuras/tabla-cadena.tex.

The chain step by step, each tournament judged by models that never saw it: of
the real hits of the four classes, the share the system detects, then detects
and classifies right (all 11 tournaments), and in VIGO, where who hit is
annotated, detects, gives to the right player and classifies right. Each step
counts only the hits that got every step before it right. From
confidence_intervals.json (scripts/evidence_confidence_intervals.py, over
chain_cv_hits.csv).

    uv run python scripts/build_chain_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import fixed, percent_range, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-cadena.tex"


def build_table() -> str:
    evidence = json.loads((METRICS / "confidence_intervals.json").read_text())
    metrics = evidence["metrics"]
    hits = evidence["params"]["chain_hits"]

    def row(label: str, key: str) -> list[str]:
        prefix = f"tabla2_cadena.{key}"
        return [
            label,
            rf"\num{{{fixed(metrics[f'{prefix}.valor'], 1, scale=100)}}}",
            percent_range(metrics[f"{prefix}.bajo"], metrics[f"{prefix}.alto"]),
        ]

    rows: list[list[str] | str] = [
        f"Los 11 torneos ({hits['four_class']} golpes)",
        row("Detecta", "detected"),
        row("Detecta y clasifica bien", "end_to_end_when_what"),
        f"Vigo, con el jugador anotado ({hits['vigo']} golpes)",
        row("Detecta", "vigo_detected"),
        row("Detecta y asigna al jugador correcto", "vigo_detected_player"),
        row("Detecta, asigna al jugador correcto y clasifica bien", "end_to_end_vigo"),
    ]
    header = ["Golpes que el sistema:", r"\%", r"Intervalo 95~\%"]
    return tabular("lcc", header, rows, rules=[3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

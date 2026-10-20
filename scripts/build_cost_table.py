"""Write the processing cost table of section 5.3: memoria/figuras/tabla-coste.tex.

Seconds per stage of `analyze_rally` on one rally, in the order of the flow,
with each stage's share of the total. From processing_time.json
(scripts/evidence_processing_time.py); the shares come from the seconds the
table prints, so they add up as the reader would add them.

    uv run python scripts/build_cost_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import percent, quantity, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-coste.tex"
STAGES = [
    ("audio_s", "Audio: detección del golpe"),
    ("pose_s", "Postura de los jugadores"),
    ("pelota_s", "Pelota"),
    ("resto_s", "Resto: lectura del vídeo, identidad, asignación y tipo"),
]


def build_table() -> str:
    metrics = json.loads((METRICS / "processing_time.json").read_text())["metrics"]
    total = metrics["total_s"]
    rows: list[list[str] | str] = [
        [label, quantity(metrics[key], 1, r"\second"), percent(metrics[key] / total)]
        for key, label in STAGES
    ]
    rows.append(["Total", quantity(total, 1, r"\second"), ""])
    return tabular("lcc", ["Etapa", "Tiempo", "Parte del total"], rows, rules=[len(STAGES)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

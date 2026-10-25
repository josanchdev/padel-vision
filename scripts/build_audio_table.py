"""Write the audio detector table of section 5.1: memoria/figuras/tabla-audio-ejecuciones.tex.

The mean of the three runs and the standard deviation of their F1, at the
threshold the system uses. From docs/metrics/audio_threshold_seeds.json, written by
scripts/evidence_audio_detector.py. That file keeps the means and the spread, not
the figures of each run, so the table has one row.

    uv run python scripts/build_audio_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.latex import number, tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-audio-ejecuciones.tex"


def build_table() -> str:
    evidence = json.loads((METRICS / "audio_threshold_seeds.json").read_text())
    metrics = evidence["metrics"]
    row = [
        "Media de tres ejecuciones",
        number(metrics["precision"], 3),
        number(metrics["recall"], 3),
        f"{number(metrics['f1'], 3)} $\\pm$ {number(metrics['f1_std'], 3)}",
    ]
    return tabular("lccc", ["Resultado", "Precisión", r"\textit{Recall}", "F1"], [row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

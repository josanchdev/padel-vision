"""Write the figures, tables and numbers of the dissertation from docs/metrics.

Reads only the evidence JSON (and the lock files, for the tools table): no GPU,
no data, a few seconds. Everything lands in memoria/figuras/generadas/, which
is wiped first, so what is there is always exactly what this script makes.

    uv run python scripts/build_thesis_assets.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from padel_ml.thesis import tables
from padel_ml.thesis.figures import draw_all
from padel_ml.thesis.latex import numbers_file
from padel_ml.thesis.numbers import thesis_numbers
from padel_ml.thesis.sources import Evidence

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "generadas"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out: Path = args.out

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    evidence = Evidence()

    (out / "cifras.tex").write_text(numbers_file(thesis_numbers(evidence)))
    for name, text in {
        "tabla-pasos.tex": tables.steps_table(evidence),
        "tabla-sistema.tex": tables.system_table(evidence),
        "tabla-coste.tex": tables.cost_table(evidence),
        "tabla-datasets.tex": tables.datasets_table(evidence),
        "tabla-herramientas.tex": tables.tools_table(
            REPO / "uv.lock", REPO / "packages" / "web" / "package-lock.json"
        ),
    }.items():
        (out / name).write_text(text + "\n")
    draw_all(evidence, out)

    for path in sorted(out.iterdir()):
        print(f"  {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

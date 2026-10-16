"""Write the adaptive window figure of section 4.2.5: memoria/figuras/ventana.pdf.

A schematic (padel_ml.thesis.window_figure): needs no data.

    uv run python scripts/build_window_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from padel_ml.thesis.window_figure import draw_window_figure

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "ventana.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    draw_window_figure(args.out)
    print(args.out)


if __name__ == "__main__":
    main()

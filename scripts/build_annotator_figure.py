"""Write the labelling tool figure of section 3.2.

Output: memoria/figuras/capturas/herramienta-etiquetado.jpg.

The annotator's window (`padel-cv annotate-types`) on a smash of a Malmö rally,
the same hit as the classifier's input figure, drawn by
padel_ml.thesis.annotator_figure with the labels saved for that rally.

    uv run python scripts/build_annotator_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from padel_ml.thesis.annotator_figure import tool_view

from padel_cv.paths import CVSPORTS_RALLIES, REPO, SHOT_TYPE_LABELS

OUT = REPO / "memoria" / "figuras" / "capturas" / "herramienta-etiquetado.jpg"
RALLY = "20231112_MALMO_02"
HIT = 615
"""A smash, as in the input figure of section 4.2.5."""
QUALITY = 95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    view = tool_view(CVSPORTS_RALLIES / f"{RALLY}.mp4", SHOT_TYPE_LABELS / f"{RALLY}.csv", HIT)
    cv2.imwrite(str(args.out), view, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    print(args.out)


if __name__ == "__main__":
    main()

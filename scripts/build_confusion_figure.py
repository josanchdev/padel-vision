"""Write the confusion matrix figure of section 5.1.4: memoria/figuras/matriz-confusion.pdf.

The shot classifier on the 2186 hits, each classified by the model trained
without its tournament, with the serve rule: shot_type_classifier.json
(scripts/evidence_shot_classifier.py), drawn by padel_ml.thesis.confusion_figure.

    uv run python scripts/build_confusion_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.confusion_figure import draw_confusion_figure

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "matriz-confusion.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    evidence = json.loads((METRICS / "shot_type_classifier.json").read_text())
    draw_confusion_figure(evidence["confusion"], evidence["labels"], args.out)
    print(args.out)


if __name__ == "__main__":
    main()

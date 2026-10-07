"""Write the problem figure of chapter 2: memoria/figuras/problema.pdf.

Needs the CVSPORTS video of the point and its cached detections
(data/cache/rally_features), but no models or GPU: the hits are those the full
system returned in its evaluation (docs/metrics/chain_cv_hits.csv).

    uv run python scripts/build_problem_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
from padel_ml.rally_features import load_rally_features
from padel_ml.thesis.problem import Crop, draw_problem_figure, system_hits

from padel_cv.arrays import ImageArray
from padel_cv.paths import CVSPORTS_RALLIES, METRICS, RALLY_FEATURES, REPO

OUT = REPO / "memoria" / "figuras" / "problema.pdf"

RALLY = "20230528_VIGO_14"
SHOWN_S = 4.52
"""J3's backhand. VIGO is the only tournament with the hitter annotated, so the
hit can be checked whole: instant, player and type are right, and so are the
four hits before and the one after it."""
CROP = Crop(x0=170, y0=180, x1=1750, y1=965)
"""The court with its glass walls; the stands and the broadcast banners are left out."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    hits = system_hits(RALLY, METRICS / "chain_cv_hits.csv")
    shown = min(hits, key=lambda hit: abs(hit.time_s - SHOWN_S))
    features = load_rally_features(RALLY_FEATURES / f"{RALLY}.npz")
    index = round(shown.time_s * features.fps)

    capture = cv2.VideoCapture(str(CVSPORTS_RALLIES / f"{RALLY}.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"{RALLY}: no se pudo leer el fotograma {index}")

    rgb = cast(ImageArray, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw_problem_figure(rgb, features.players[index], hits, shown, CROP, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

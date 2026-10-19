"""Write the court marking figure of section 4.1.

Output: memoria/figuras/capturas/marcado-pista.jpg.

The Madrid court, on the frame it was marked on: the six points saved in its
court file and the court lines their homography projects, drawn by
padel_ml.thesis.court_marking_figure.

    uv run python scripts/build_court_marking_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from padel_ml.thesis.court_marking_figure import load_marked_court, marking_view

from padel_cv.paths import COURTS, REPO
from padel_cv.video_io import read_frame

OUT = REPO / "memoria" / "figuras" / "capturas" / "marcado-pista.jpg"
COURT = COURTS / "20230927_MADRID.json"
"""Madrid: the same tournament as the players figure of section 4.2.2."""
QUALITY = 95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    court = load_marked_court(COURT)
    view = marking_view(read_frame(court.video, court.frame_index), court.points_px)
    cv2.imwrite(str(args.out), view, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    print(args.out)


if __name__ == "__main__":
    main()

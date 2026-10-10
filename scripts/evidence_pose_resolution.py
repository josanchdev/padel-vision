"""How the pose model's input size changes who it finds (the 1920 px of section 4.2.2).

YOLO shrinks every frame to `imgsz` before detecting. Far-side players are about
60 px tall in a 1920x1080 frame, so shrinking to the default 640 leaves them
about 20 px. This measures it: for one rally of every tournament, the pose model
runs at several input sizes with the system's own threshold, the people are kept
by the court mask (as in the system), and per frame it records

  - whether the four players are in the mask (`all_four`),
  - whether both far-side players are (`far_pair`; far side = above the net row),
  - the time per frame (detector alone, no tracker).

Every `--stride`-th frame is used, for speed.

    uv run python scripts/evidence_pose_resolution.py [--stride 4]
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from padel_ml.evidence import ExperimentResult
from padel_ml.rally_analysis import POSE_CONFIDENCE

from padel_cv.court_registry import court_file_for, load_corners, load_homography
from padel_cv.paths import CVSPORTS_RALLIES
from padel_cv.player_identity import court_mask_polygon, filter_players
from padel_cv.pose import PoseDetector

SIZES = (640, 960, 1280, 1920)
RALLY_INDEX = "01"
"""The second rally of every tournament."""
NET_CENTRE_M = (5.0, 10.0)


def net_row_px(homography: np.ndarray) -> float:
    """Image row of the net's centre on the ground: far-side players are above it."""
    point = np.linalg.inv(homography) @ np.array([*NET_CENTRE_M, 1.0])
    return float(point[1] / point[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    rallies = sorted(CVSPORTS_RALLIES.glob(f"*_{RALLY_INDEX}.mp4"))
    totals = {size: {"frames": 0, "all_four": 0, "far_pair": 0, "seconds": 0.0} for size in SIZES}
    used: list[str] = []
    for video in rallies:
        court = court_file_for(video.stem)
        if court is None:
            continue
        used.append(video.stem)
        polygon = court_mask_polygon(load_corners(court))
        net_row = net_row_px(load_homography(court))
        for size in SIZES:
            detector = PoseDetector(confidence=POSE_CONFIDENCE, image_size=size, tracker=None)
            capture = cv2.VideoCapture(str(video))
            index = 0
            while True:
                ok, grabbed = capture.read()
                if not ok:
                    break
                frame = np.asarray(grabbed, dtype=np.uint8)
                index += 1
                if (index - 1) % args.stride:
                    continue
                if index == 1:
                    detector.detect(frame)  # warm-up, not counted
                start = time.perf_counter()
                poses = detector.detect(frame)
                elapsed = time.perf_counter() - start
                players = filter_players(poses, polygon)
                far = [p for p in players if p.bbox_xyxy[3] < net_row]
                row = totals[size]
                row["frames"] += 1
                row["all_four"] += len(players) >= 4
                row["far_pair"] += len(far) >= 2
                row["seconds"] += elapsed
            capture.release()
        print(f"{video.stem}: hecho")

    metrics: dict[str, float] = {}
    for size, row in totals.items():
        n = row["frames"]
        metrics[f"all_four_{size}"] = round(row["all_four"] / n, 4)
        metrics[f"far_pair_{size}"] = round(row["far_pair"] / n, 4)
        metrics[f"ms_per_frame_{size}"] = round(1000 * row["seconds"] / n, 1)
        print(
            f"  {size:5d} px  cuatro {row['all_four'] / n:6.1%}  "
            f"pareja del fondo {row['far_pair'] / n:6.1%}  {1000 * row['seconds'] / n:5.1f} ms"
        )

    result = ExperimentResult(
        name="pose_resolution",
        summary=(
            "Tamaño de entrada del modelo de pose: fotogramas con los cuatro jugadores "
            f"dentro de la máscara a {', '.join(str(s) for s in SIZES)} px: "
            + ", ".join(f"{metrics[f'all_four_{s}']:.0%}" for s in SIZES)
            + "."
        ),
        metrics=metrics,
        dataset=(
            f"{len(used)} puntos (el «{RALLY_INDEX}» de cada torneo), 1 de cada {args.stride} "
            f"fotogramas ({totals[SIZES[0]]['frames']} fotogramas)"
        ),
        method="YOLO26n-pose sin seguimiento, umbral del sistema, máscara de pista del sistema",
        params={"sizes": list(SIZES), "confidence": POSE_CONFIDENCE, "stride": args.stride},
        notes=(
            "«cuatro» = al menos cuatro personas con la base de la caja dentro de la máscara. "
            "«pareja del fondo» = al menos dos de ellas por encima de la red. "
            "Un fotograma con cuatro no garantiza que sean los cuatro jugadores, pero la máscara "
            "ya descarta al público."
        ),
    )
    print(f"guardado -> {result.save()}")


if __name__ == "__main__":
    main()

"""Evidence for the ball detector as the system runs it (ADR-0009).

Training validated TrackNet on cached 512x288 frames and counted a detection as
correct within 4 cells of its 128x72 output grid — about 60 px in 1080p, for a
ball that is ~8 px across. The system, though, runs the detector at 768x432
(`BallDetector`) and cleans the track afterwards (`postprocess_ball`). This
measures exactly that, on the match it never trained on, in real 1080p pixels:

    data     PadelTracker100, women's final (training used the men's final)
    truth    the dataset's ball box centre, every frame labelled
    scored   per frame, the TrackNet way: TP if the ball is found within the
             tolerance; a ball found too far away or where there is none is a
             false positive; a missed ball is a false negative

Tolerances fixed before measuring: 15 px (one output cell), 30 px, and 60 px
(the training criterion, for continuity with the training log).

    uv run python scripts/evidence_ball_detector.py
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
from padel_ml.ball_infer import BallDetector, BallHit
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.evidence import ExperimentResult

from padel_cv.ball_data import load_ball_centers

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data" / "raw" / "2022_BCN_FinalF_1.mp4"
LABELS = REPO / "data" / "labels" / "2022_BCN_FinalF_1_ball.json"
CHECKPOINT = REPO / "runs" / "ball_full" / "tracknetv3.pt"
TOLERANCES_PX = (15, 30, 60)
HEADLINE_PX = 15


def score(
    found: dict[int, tuple[float, float]],
    truth: dict[int, tuple[float, float] | None],
    tolerance_px: float,
) -> dict[str, float]:
    """Precision, recall and F1 over every labelled frame, plus the error of the hits."""
    tp = fp = fn = 0
    errors: list[float] = []
    for frame, ball in truth.items():
        guess = found.get(frame)
        if ball is None:
            fp += int(guess is not None)
            continue
        if guess is None:
            fn += 1
            continue
        error = float(((guess[0] - ball[0]) ** 2 + (guess[1] - ball[1]) ** 2) ** 0.5)
        if error <= tolerance_px:
            tp += 1
            errors.append(error)
        else:
            fp += 1  # found, but somewhere else
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "median_error_px": round(statistics.median(errors), 2) if errors else 0.0,
    }


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    centers = load_ball_centers(LABELS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Every frame of this match is labelled; a frame with no box has no ball.
    truth: dict[int, tuple[float, float] | None] = {i: None for i in range(total)}
    for frame, ball in centers.items():
        if ball.center_xy is not None:
            truth[frame] = (ball.center_xy[0] * width, ball.center_xy[1] * height)

    detector = BallDetector(CHECKPOINT)
    raw: list[BallHit] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        hit = detector.detect(image, index)
        if hit is not None:
            raw.append(hit)
        index += 1
        if index % 5000 == 0:
            print(f"  {index}/{total} frames", flush=True)
    capture.release()

    tracks = {
        "detector": {h.frame_index: (h.x_px, h.y_px) for h in raw},
        "detector + limpieza": {h.frame_index: (h.x_px, h.y_px) for h in postprocess_ball(raw)},
    }
    results = {
        name: {f"{tol}px": score(track, truth, tol) for tol in TOLERANCES_PX}
        for name, track in tracks.items()
    }
    for name, by_tolerance in results.items():
        for tolerance, row in by_tolerance.items():
            print(f"{name:22s} {tolerance:>5s}  {row}")

    headline = results["detector"][f"{HEADLINE_PX}px"]
    with_ball = sum(1 for ball in truth.values() if ball is not None)
    result = ExperimentResult(
        name="ball_detector",
        summary=(
            f"Detector de pelota (TrackNetV3 propio) tal como lo usa el sistema, en un partido "
            f"no visto: F1 {headline['f1']:.3f} a {HEADLINE_PX} px en 1080p "
            f"(precisión {headline['precision']:.3f}, recall {headline['recall']:.3f})."
        ),
        metrics={
            "f1": headline["f1"],
            "precision": headline["precision"],
            "recall": headline["recall"],
            "median_error_px": headline["median_error_px"],
            "tolerance_px": HEADLINE_PX,
        },
        dataset=(
            f"PadelTracker100, final femenina 2022 ({total} frames, {with_ball} con pelota). "
            "El modelo se entrenó con la final masculina."
        ),
        method="TrackNetV3 entrenado en pádel, inferencia a 768x432, umbral 0,5",
        params={
            "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "input_size": [768, 432],
            "threshold": 0.5,
            "tolerances_px": list(TOLERANCES_PX),
            "by_tolerance": results,
        },
        notes=(
            "Se mide el detector tal como corre en el sistema, no como se validó al entrenar "
            "(frames precortados a 512x288 y 4 celdas de tolerancia, unos 60 px en 1080p, que "
            "dio F1 0,942). La limpieza de trayectoria es la que usa el voto de quién golpea."
        ),
    )
    print(f"guardado -> {result.save()}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()

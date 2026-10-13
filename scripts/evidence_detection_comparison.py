"""Detecting hits from the image or from the sound, on the same five points (section 4.2.1).

The image-based detector (a model over pose and ball, ADR-0013) was measured once,
before it was retired, on five VIGO points it did not train on: 20230528_VIGO_00,
02, 03, 10 and 11, the validation side of a 70/30 split of the 17 VIGO points
(seed 0). Its result is in method_comparison_same_data.json, and its code stays
under the git tag `pre-limpieza`.

That file set it against an audio F1 copied from another evaluation, over all 99
points. This script puts the audio on the same five points instead, read from the
full-chain evaluation (chain_cv_hits.csv), where the audio detector of the VIGO
round was trained without any VIGO point. Same five points, same 94 real hits,
same rule: a detection counts if it falls within 250 ms of a real hit, one to one.

Time is measured on those five points too, as seconds of processing per minute
of video on this machine: for the audio, the detector the system runs; for the
image, extracting the pose and the ball it needs, frame by frame, as the system
does. The retired model on top of them is not counted (its code is no longer in
the system), so the image figure is a lower bound.

    uv run python scripts/evidence_detection_comparison.py
"""

from __future__ import annotations

import csv
import json
import time

import cv2
import numpy as np
import torch
from padel_ml.audio_train import detect_hits_in_audio
from padel_ml.ball_infer import BallDetector
from padel_ml.evidence import ExperimentResult
from padel_ml.rally_analysis import POSE_CONFIDENCE

from padel_cv.court_registry import court_file_for, load_corners
from padel_cv.paths import CVSPORTS_RALLIES, METRICS, RUNS
from padel_cv.player_identity import court_mask_polygon, filter_players
from padel_cv.pose import PoseDetector

TOURNAMENT = "20230528_VIGO"
SPLIT_SEED = 0
VALIDATION_SHARE = 0.3
"""How the image detector's measurement split the VIGO points (pre-limpieza)."""


def validation_points() -> list[str]:
    """The five VIGO points the image detector was measured on."""
    names = [p.stem for p in sorted(CVSPORTS_RALLIES.glob(f"{TOURNAMENT}_*.mp4"))]
    order = np.random.default_rng(SPLIT_SEED).permutation(len(names))
    n_val = max(int(len(names) * VALIDATION_SHARE), 1)
    return sorted(names[i] for i in order[:n_val])


def audio_counts(points: list[str]) -> dict[str, int]:
    """Matched, false and missed hits of the audio on `points`, from the chain run."""
    counts = {"tp": 0, "fp": 0, "fn": 0}
    with (METRICS / "chain_cv_hits.csv").open() as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row["rally"] not in points:
                continue
            if not row["truth_s"]:
                counts["fp"] += 1
            elif not row["detected_s"]:
                counts["fn"] += 1
            else:
                counts["tp"] += 1
    return counts


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4),
    }


def video_minutes(points: list[str]) -> float:
    total = 0.0
    for point in points:
        capture = cv2.VideoCapture(str(CVSPORTS_RALLIES / f"{point}.mp4"))
        total += capture.get(cv2.CAP_PROP_FRAME_COUNT) / (capture.get(cv2.CAP_PROP_FPS) or 25.0)
        capture.release()
    return total / 60.0


def time_audio(points: list[str], device: str) -> float:
    """Seconds the system's audio detector takes over `points`."""
    checkpoint = RUNS / "audio" / "audio_crnn.pt"
    detect_hits_in_audio(CVSPORTS_RALLIES / f"{points[0]}.mp4", checkpoint, device=device)
    start = time.perf_counter()
    for point in points:
        detect_hits_in_audio(CVSPORTS_RALLIES / f"{point}.mp4", checkpoint, device=device)
    return time.perf_counter() - start


def time_pose_and_ball(points: list[str], device: str) -> float:
    """Seconds to extract, frame by frame, the pose and the ball the image detector reads."""
    pose = PoseDetector(confidence=POSE_CONFIDENCE, device=device)
    ball = BallDetector(RUNS / "ball_full" / "tracknetv3.pt", device=device)
    total = 0.0
    for number, point in enumerate([points[0], *points]):
        court = court_file_for(point)
        polygon = court_mask_polygon(load_corners(court)) if court is not None else None
        pose.reset()
        ball.reset()
        capture = cv2.VideoCapture(str(CVSPORTS_RALLIES / f"{point}.mp4"))
        start = time.perf_counter()
        index = 0
        while True:
            ok, grabbed = capture.read()
            if not ok:
                break
            frame = np.asarray(grabbed, dtype=np.uint8)
            filter_players(pose.detect(frame), polygon)
            ball.detect(frame, index)
            index += 1
        elapsed = time.perf_counter() - start
        capture.release()
        if number > 0:  # the first pass only warms the GPU up
            total += elapsed
    return total


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = validation_points()
    image = json.loads((METRICS / "method_comparison_sweep.json").read_text())
    image_counts = {key: int(image[key]) for key in ("tp", "fp", "fn")}
    audio = audio_counts(points)
    if image_counts["tp"] + image_counts["fn"] != audio["tp"] + audio["fn"]:
        raise SystemExit(f"distinto número de golpes reales: {image_counts} y {audio}")

    minutes = video_minutes(points)
    audio_s = time_audio(points, device)
    image_s = time_pose_and_ball(points, device)

    metrics = {
        **{f"image_{k}": v for k, v in scores(**image_counts).items()},
        **{f"audio_{k}": v for k, v in scores(**audio).items()},
        "real_hits": audio["tp"] + audio["fn"],
        "video_minutes": round(minutes, 3),
        "audio_s_per_min": round(audio_s / minutes, 2),
        "image_s_per_min": round(image_s / minutes, 2),
    }
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    ExperimentResult(
        name="detection_comparison",
        summary=(
            f"Detección de golpes en los mismos {len(points)} puntos de VIGO: imagen F1 "
            f"{metrics['image_f1']:.3f}, audio F1 {metrics['audio_f1']:.3f}; "
            f"{metrics['image_s_per_min']:.1f} s frente a {metrics['audio_s_per_min']:.2f} s "
            "por minuto de vídeo."
        ),
        metrics=metrics,
        dataset=f"CVSPORTS_Padel: {', '.join(points)} ({metrics['real_hits']} golpes reales)",
        method=(
            "Imagen: modelo sobre pose y pelota (ADR-0013), medido antes de retirarlo "
            "(method_comparison_sweep.json). Audio: el detector de la ronda VIGO de la cadena "
            "completa, entrenado sin ningún punto de VIGO (chain_cv_hits.csv)."
        ),
        params={
            "points": points,
            "collar_s": 0.25,
            "image_counts": image_counts,
            "audio_counts": audio,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        },
        notes=(
            "Mismos puntos, mismos golpes reales y misma regla de emparejamiento (250 ms, "
            "uno a uno). El modelo de imagen se entrenó con los otros 12 puntos de VIGO y su "
            "umbral se eligió sobre estos cinco, lo que le favorece; el audio, con los otros "
            "10 torneos. El tiempo de la imagen solo cuenta extraer pose y pelota, sin el "
            "modelo retirado, así que es una cota inferior."
        ),
    ).save()


if __name__ == "__main__":
    main()

"""External evaluation: the full system on footage it has never seen, scored
against Jorge's blind labels (see `padel_ml.external_eval`).

Every clip with labels in data/labels/external/ is run end to end with the
committed code — the system is frozen before labelling, and the commit is stored
with the result so the number can be traced to the exact code that produced it.

    uv run python scripts/evaluate_external.py
    uv run python scripts/evaluate_external.py miami_rally1   # a subset

Writes docs/metrics/external_evaluation.json, docs/metrics/external_hits.csv
(every hit, the table behind every number) and the type confusion figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.evidence import (
    FIGURES_DIR,
    METRICS_DIR,
    ExperimentResult,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    plot_confusion,
)
from padel_ml.external_eval import (
    COLLAR_S,
    FREEZE_MAX_CHANGED_PX,
    FREEZE_MIN_FRAMES,
    MAX_CLICK_DISTANCE,
    HitRow,
    frozen_frames,
    load_truth,
    other_hits,
    score_clip,
    summarize,
    type_confusion,
    write_rows,
)
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally

from padel_cv.court_registry import court_file_for

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "data" / "labels" / "external"
VIDEOS = REPO / "data" / "raw" / "external"

COUNTS = {
    "labelled_hits",
    "detected_hits",
    "hitter_not_detected",
    "in_freeze",
    "detections_in_freeze_unjudged",
}

SPANISH = {
    "labelled_hits": "golpes etiquetados",
    "detected_hits": "golpes detectados",
    "detection_precision": "deteccion: precision",
    "detection_recall": "deteccion: recall",
    "detection_f1": "deteccion: F1",
    "player_accuracy": "quien: jugador",
    "team_accuracy": "quien: equipo",
    "hitter_not_detected": "golpeador sin detectar",
    "type_accuracy": "tipo (golpes detectados)",
    "type_accuracy_right_player": "tipo (con el jugador correcto)",
    "end_to_end": "extremo a extremo",
    "in_freeze": "golpes en congelacion",
    "detections_in_freeze_unjudged": "detecciones en congelacion (sin juzgar)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", help="Clip names (default: every labelled clip)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    clips = args.clips or sorted(p.stem for p in LABELS.glob("*.csv"))
    if not clips:
        parser.error(f"No labels in {LABELS}")
    device = args.device if torch.cuda.is_available() else "cpu"
    models = RallyModels(ModelPaths.under(REPO / "runs"), device=device)

    rows: list[HitRow] = []
    per_clip: dict[str, dict[str, float]] = {}
    for clip in clips:
        video = VIDEOS / f"{clip}.mp4"
        court = court_file_for(clip)
        if court is None:  # never run external footage without its court
            parser.error(f"{clip}: no court marked (padel-cv annotate-court)")
        print(f"[{clip}] analizando...", flush=True)
        analysis = analyze_rally(video, models, court)
        clip_rows = score_clip(
            clip, load_truth(LABELS / f"{clip}.csv"), analysis, frozen_frames(video)
        )
        per_clip[clip] = summarize(clip_rows)
        rows.extend(clip_rows)

    overall = summarize(rows)
    truth, predicted, labels = type_confusion(rows)
    matrix = confusion_matrix(truth, predicted, labels)
    per_class = per_class_metrics(matrix, labels)
    per_class.pop("Unclassified", None)  # a column for abstentions, not a class
    metrics: dict[str, float] = {**overall, "type_macro_f1": macro_f1(per_class)}
    for clip, numbers in per_clip.items():
        metrics.update({f"{clip}/{k}": v for k, v in numbers.items()})

    result = ExperimentResult(
        name="external_evaluation",
        summary=(
            f"Sistema completo sobre metraje externo ({overall['labelled_hits']:.0f} golpes, "
            f"etiquetado a ciegas): deteccion F1 {overall['detection_f1']:.3f}, jugador "
            f"{overall['player_accuracy']:.1%}, tipo {overall['type_accuracy']:.1%}, extremo a "
            f"extremo {overall['end_to_end']:.1%}."
        ),
        metrics=metrics,
        dataset=f"Clips externos a CVSPORTS, etiquetados a ciegas: {', '.join(clips)}",
        method=(
            "analyze_rally: audio CRNN + YOLO26-pose + TrackNetV3 + voto +-4 + BST-0 "
            "+ regla del saque"
        ),
        params={
            "collar_s": COLLAR_S,
            "max_click_distance_body_heights": MAX_CLICK_DISTANCE,
            "freeze_min_frames": FREEZE_MIN_FRAMES,
            "freeze_max_changed_px": FREEZE_MAX_CHANGED_PX,
            "clips": clips,
            "other_labelled_as": other_hits(rows),
        },
        per_class=per_class,
        confusion=matrix.tolist(),
        labels=labels,
        notes=(
            "Emparejamiento: cada golpe del sistema con el etiquetado mas cercano a <=250 ms "
            "(mismo criterio que el detector de audio y el paper). Jugador: el clic del "
            "etiquetado se resuelve al jugador detectado que esta ahi, asi que no depende de "
            "que el sistema mantenga J1-J4. Equipo: mismo lado de la red, medido en la pista. "
            "Tipo sobre golpes emparejados con etiqueta de las cuatro clases; 'Other' cuenta "
            "para cuando y quien pero no para tipo. Condiciones de cada clip en "
            "docs/experiments.md."
        ),
    )
    path = result.save()
    write_rows(rows, METRICS_DIR / "external_hits.csv")
    plot_confusion(
        matrix,
        labels,
        f"Tipo de golpe, metraje externo ({len(truth)} golpes)",
        FIGURES_DIR / "external_type_confusion.png",
    )

    print()
    for key, value in overall.items():
        shown = f"{value:g}" if key in COUNTS else f"{value:.1%}"
        print(f"  {SPANISH.get(key, key):32s} {shown}")
    print(f"  {'tipo: macro-F1':32s} {metrics['type_macro_f1']:.3f}")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()

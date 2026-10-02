"""Regenerate the hit-assignment evidence (ADR-0015 D) into docs/metrics/.

Measures our replica of Decorte 2024 §5.3 against their published ground truth
(319 hits) and saves: per-rally table, confusion matrix, per-class metrics and
the figures — so the dissertation cites files, not remembered numbers. The court
is the hand-marked one each rally uses in production (ADR-0018).

    uv run python scripts/evidence_hit_assignment.py
"""

from __future__ import annotations

import argparse
import collections
import json

from padel_ml.ball_infer import BallDetector
from padel_ml.evidence import (
    FIGURES_DIR,
    ExperimentResult,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    plot_comparison,
    plot_confusion,
)
from padel_ml.hit_assignment import WINDOW_HALF, assign_hit, team_alternation_sweep
from padel_ml.hit_assignment_eval import annotated_rallies
from padel_ml.rally_analysis import POSE_CONFIDENCE

from padel_cv.paths import COURTS, CVSPORTS, METRICS, RUNS
from padel_cv.pose import PoseDetector

PAPER_PLAYER, PAPER_TEAM = 0.8370, 0.8683
LABELS = ["J1", "J2", "J3", "J4", "sin asignar"]


def main() -> None:
    pose_stage = PoseDetector(confidence=POSE_CONFIDENCE)  # as analyze_rally
    ball_detector = BallDetector(RUNS / "ball_full" / "tracknetv3.pt")

    true_labels: list[str] = []
    predicted: list[str | None] = []
    per_rally: list[dict[str, float | int | str]] = []

    for annotated in annotated_rallies(CVSPORTS, COURTS, pose_stage, ball_detector):
        rally, fps, states = annotated.name, annotated.fps, annotated.states
        predictions: dict[int, int | None] = {}
        hit_of: dict[int, object] = {}
        for hit in annotated.hits:
            frame = round(hit.time_s * fps)
            predictions[frame] = assign_hit(frame, states)
            hit_of[frame] = hit
        predictions = team_alternation_sweep(predictions)

        player_ok = team_ok = 0
        for frame, prediction in predictions.items():
            hit = hit_of[frame]
            true_labels.append(f"J{hit.slot}")  # type: ignore[attr-defined]
            if prediction is None:
                predicted.append(None)
            elif prediction < 0:  # team-only recovery
                predicted.append(None)
                team_ok += int(-prediction == hit.team)  # type: ignore[attr-defined]
            else:
                predicted.append(f"J{prediction}")
                player_ok += int(prediction == hit.slot)  # type: ignore[attr-defined]
                team_ok += int((1 if prediction <= 2 else 2) == hit.team)  # type: ignore[attr-defined]
        total = len(predictions)
        per_rally.append(
            {
                "rally": rally,
                "hits": total,
                "player_accuracy": round(player_ok / total, 4),
                "team_accuracy": round(team_ok / total, 4),
            }
        )
        print(f"{rally}: jugador {player_ok / total:.1%} equipo {team_ok / total:.1%} ({total})")

    errors = collections.Counter(
        (t, p or "sin asignar") for t, p in zip(true_labels, predicted, strict=True) if t != p
    )
    top_errors = ", ".join(f"{t}->{p} ({n})" for (t, p), n in errors.most_common(3))
    matrix = confusion_matrix(true_labels, predicted, LABELS)
    per_class = per_class_metrics(matrix, LABELS)
    total_hits = sum(int(r["hits"]) for r in per_rally)
    player_acc = sum(r["player_accuracy"] * r["hits"] for r in per_rally) / total_hits  # type: ignore[operator]
    team_acc = sum(r["team_accuracy"] * r["hits"] for r in per_rally) / total_hits  # type: ignore[operator]

    plot_confusion(
        matrix,
        LABELS,
        "Asignación de golpe a jugador (319 golpes, VIGO)",
        FIGURES_DIR / "hit_assignment_confusion.png",
    )
    plot_comparison(
        {
            "Nuestra réplica": {"accuracy": round(player_acc, 4)},
            "Paper (Decorte 2024)": {"accuracy": PAPER_PLAYER},
        },
        "accuracy",
        "Asignación por jugador vs el paper",
        FIGURES_DIR / "hit_assignment_vs_paper.png",
        reference=("Paper: 83,70%", PAPER_PLAYER),
    )

    result = ExperimentResult(
        name="hit_assignment_replica",
        summary=(
            f"Réplica del 'quién golpea' de Decorte 2024: jugador {player_acc:.2%} "
            f"(paper {PAPER_PLAYER:.2%}), equipo {team_acc:.2%} (paper {PAPER_TEAM:.2%})."
        ),
        metrics={
            "player_accuracy": round(player_acc, 4),
            "team_accuracy": round(team_acc, 4),
            "macro_f1": macro_f1(per_class),
            "paper_player_accuracy": PAPER_PLAYER,
            "paper_team_accuracy": PAPER_TEAM,
            "hits": total_hits,
        },
        dataset="CVSPORTS_Padel — 319 golpes anotados con jugador (16 rallies VIGO)",
        method="Voto ponderado multi-frame (ec.1) + fallbacks + barrido por alternancia",
        params={
            "window_half_frames": WINDOW_HALF,
            "pose_confidence": POSE_CONFIDENCE,
            "min_keypoint_confidence": 0.3,
            "court": "marcada a mano, 20230528_VIGO.json (ADR-0018)",
            "ball": "TrackNetV3 propio a 768x432 + post-proceso (ball_detector.json)",
        },
        per_class=per_class,
        confusion=matrix.tolist(),
        labels=LABELS,
        notes=(
            "Se evalúa con los instantes ANOTADOS, no con los detectados por audio, para "
            "medir la asignación aislada. Mismos 319 golpes y misma métrica que la tabla 3 "
            "del paper: acierto ponderado por golpes, y los golpes sin asignar cuentan como "
            "fallo en ambos. Las mejoras (distancia normalizada por la altura del jugador, "
            "ventana de ±4 frames, resolución de la pelota) se eligieron midiendo sobre estos "
            "mismos golpes, así que la cifra puede ser algo optimista. Como en el paper, "
            "un barrido por alternancia recupera el EQUIPO de los golpes sin jugador; el "
            "sistema completo no lo aplica (tabla 2). Errores más "
            f"frecuentes: {top_errors}."
        ),
    )
    path = result.save()
    (METRICS / "hit_assignment_per_rally.json").write_text(json.dumps(per_rally, indent=2) + "\n")
    print(f"\nGLOBAL jugador {player_acc:.2%} equipo {team_acc:.2%}")
    print("errores mas frecuentes:")
    for (truth, prediction), count in errors.most_common(6):
        print(f"  {truth} -> {prediction}: {count}")
    print(f"guardado -> {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()

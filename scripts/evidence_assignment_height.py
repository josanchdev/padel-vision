"""What dividing by the player's height does to the hit assignment (section 4.2.4).

The vote measures, in every frame of the window, the distance from the ball to
each player's closer wrist divided by the height of the player's box
(`padel_ml.hit_assignment.player_distance`). This scores the same vote twice on
the 319 hits the dataset's paper annotated with who hit (16 VIGO points): as the
system runs it, and with the plain pixel distance instead, everything else equal.

The system's code is not changed: for the second run this script swaps the
distance function the vote looks up, only while it runs. Shot types come from
our own labels, through the full-chain evaluation (chain_cv_hits.csv), which
already paired each annotated VIGO hit with its type.

    uv run python scripts/evidence_assignment_height.py
"""

from __future__ import annotations

import csv
from collections.abc import Callable

import numpy as np
from padel_ml import hit_assignment
from padel_ml.ball_infer import BallDetector
from padel_ml.evidence import ExperimentResult
from padel_ml.hit_assignment import assign_hit
from padel_ml.hit_assignment_eval import annotated_rallies
from padel_ml.rally_analysis import POSE_CONFIDENCE

from padel_cv.detections import PoseDetection
from padel_cv.paths import COURTS, CVSPORTS, METRICS, RUNS
from padel_cv.pose import PoseDetector

TYPE_MATCH_S = 0.15
"""How close an annotated hit and a typed hit must be to be the same hit."""

Distance = Callable[[PoseDetection, tuple[float, float]], float | None]


def pixel_distance(pose: PoseDetection, ball_xy: tuple[float, float]) -> float | None:
    """The system's distance before dividing by the height: plain pixels."""
    heights = SYSTEM_DISTANCE(pose, ball_xy)
    if heights is None:
        return None
    return heights * max(pose.bbox_xyxy[3] - pose.bbox_xyxy[1], 1.0)


SYSTEM_DISTANCE: Distance = hit_assignment.player_distance


def shot_types() -> dict[str, list[tuple[float, str]]]:
    """(time, type) of every typed VIGO hit, per point."""
    out: dict[str, list[tuple[float, str]]] = {}
    with (METRICS / "chain_cv_hits.csv").open() as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row["tournament"] == "20230528_VIGO" and row["truth_s"] and row["truth_type"]:
                out.setdefault(row["rally"], []).append((float(row["truth_s"]), row["truth_type"]))
    return out


def type_of(types: list[tuple[float, str]], time_s: float) -> str | None:
    if not types:
        return None
    nearest = min(types, key=lambda t: abs(t[0] - time_s))
    return nearest[1] if abs(nearest[0] - time_s) <= TYPE_MATCH_S else None


def accuracy(rows: list[dict[str, object]], key: str, only: str | None = None) -> tuple[int, int]:
    """(correct, total) of `key` predictions, optionally for one shot type."""
    chosen = [r for r in rows if only is None or r["type"] == only]
    return sum(r[key] == r["truth"] for r in chosen), len(chosen)


def main() -> None:
    pose = PoseDetector(confidence=POSE_CONFIDENCE)
    ball = BallDetector(RUNS / "ball_full" / "tracknetv3.pt")
    types = shot_types()
    rows: list[dict[str, object]] = []
    for annotated in annotated_rallies(CVSPORTS, COURTS, pose, ball):
        rally = annotated.name
        for hit in annotated.hits:
            frame = round(hit.time_s * annotated.fps)
            setattr(hit_assignment, "player_distance", SYSTEM_DISTANCE)  # noqa: B010
            heights = assign_hit(frame, annotated.states)
            setattr(hit_assignment, "player_distance", pixel_distance)  # noqa: B010
            pixels = assign_hit(frame, annotated.states)
            setattr(hit_assignment, "player_distance", SYSTEM_DISTANCE)  # noqa: B010
            rows.append(
                {
                    "rally": rally,
                    "truth": hit.slot,
                    "type": type_of(types.get(rally, []), hit.time_s),
                    "heights": heights,
                    "pixels": pixels,
                }
            )
        print(f"{rally}: {len(annotated.hits)} golpes", flush=True)

    metrics: dict[str, float] = {"hits": len(rows)}
    for group in (None, "Smash", "Forehand", "Backhand", "Serve"):
        name = (group or "all").lower()
        ok_h, n = accuracy(rows, "heights", group)
        ok_p, _ = accuracy(rows, "pixels", group)
        if n == 0:
            continue
        metrics[f"{name}_n"] = n
        metrics[f"{name}_pixels"] = round(ok_p / n, 4)
        metrics[f"{name}_heights"] = round(ok_h / n, 4)
        metrics[f"{name}_errors_pixels"] = n - ok_p
        metrics[f"{name}_errors_heights"] = n - ok_h
    fixed = sum(r["pixels"] != r["truth"] and r["heights"] == r["truth"] for r in rows)
    broken = sum(r["pixels"] == r["truth"] and r["heights"] != r["truth"] for r in rows)
    metrics["fixed"] = int(fixed)
    metrics["broken"] = int(broken)
    smash_errors = metrics["smash_errors_pixels"]
    metrics["smash_errors_resolved"] = (
        round((smash_errors - metrics["smash_errors_heights"]) / smash_errors, 4)
        if smash_errors
        else float(np.nan)
    )
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    ExperimentResult(
        name="assignment_height",
        summary=(
            f"Asignación del golpe con la distancia dividida por la altura frente a en píxeles: "
            f"jugador {metrics['all_heights']:.1%} frente a {metrics['all_pixels']:.1%}; "
            f"remates {metrics['smash_heights']:.1%} frente a {metrics['smash_pixels']:.1%}."
        ),
        metrics=metrics,
        dataset="CVSPORTS_Padel VIGO: los 319 golpes con el jugador anotado (16 puntos)",
        method=(
            "El voto del sistema (padel_ml.hit_assignment.assign_hit), una vez con su distancia "
            "y otra con la misma distancia sin dividir por la altura de la caja"
        ),
        params={"window_half": hit_assignment.WINDOW_HALF, "type_match_s": TYPE_MATCH_S},
        notes=(
            "Instantes anotados (no los del audio), para medir la asignación aislada. Sin el "
            "barrido por alternancia, que el sistema no aplica: un golpe sin asignar cuenta "
            "como fallo. Tipos de golpe de nuestro etiquetado, emparejados a 0,15 s."
        ),
    ).save()


if __name__ == "__main__":
    main()

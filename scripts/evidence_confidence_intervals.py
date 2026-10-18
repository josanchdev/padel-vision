"""95% confidence intervals for the headline numbers, by resampling rallies.

A figure measured on 319 hits from 16 rallies carries sampling uncertainty, and
the hits of one rally are not independent (same players, same court, the same
identity tracking). So the resampling unit is the rally, not the hit — a cluster
bootstrap: draw rallies with replacement, recompute the metric with the code that
produced it, repeat, and keep the 2.5 and 97.5 percentiles. The shot-type
cross-validation keeps its results per tournament, so there the unit is the
tournament (11 of them: a wider, more cautious interval).

The comparison with the paper is PAIRED: both methods were scored on the same
16 rallies (their Table 3 gives each one), so the rallies are resampled once and
the difference is taken on the same draw — each rally's difficulty cancels out.

The whole chain is also given step by step (section 5.2 of the dissertation): of
the real hits of the four classes, the share detected, then detected and given
to the right player (VIGO), with the share that gets every step right.

Reads the evidence files and the label cache; no GPU, no models.

    uv run python scripts/evidence_confidence_intervals.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from padel_ml.chain_eval import Row, ratio, summarize
from padel_ml.evidence import ExperimentResult
from padel_ml.hit_assignment_gt import PAPER_TABLE_3
from padel_ml.shot_type_dataset import CLASSES, build_dataset

from padel_cv.paths import METRICS, RALLY_FEATURES, SHOT_TYPE_LABELS

RESAMPLES = 10_000
SEED = 0

T = TypeVar("T")


def bootstrap(
    clusters: Sequence[T], metrics: Callable[[list[T]], dict[str, float]]
) -> dict[str, list[float]]:
    """[point estimate, low, high] for every metric, resampling whole clusters."""
    rng = np.random.default_rng(SEED)
    point = metrics(list(clusters))
    draws: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(RESAMPLES):
        sample = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
        for key, value in metrics(sample).items():
            draws[key].append(value)
    intervals = {}
    for key in point:
        low, high = np.percentile(draws[key], [2.5, 97.5])
        intervals[key] = [round(point[key], 4), round(float(low), 4), round(float(high), 4)]
    return intervals


def assignment() -> dict[str, list[float]]:
    """Table 1, who hit: ours, and ours minus the paper's, on the same 16 rallies."""
    rallies = json.loads((METRICS / "hit_assignment_per_rally.json").read_text())
    clusters = []
    for ours, (paper_player, paper_team, hits) in zip(rallies, PAPER_TABLE_3, strict=True):
        if ours["hits"] != hits:
            raise SystemExit(f"{ours['rally']}: {ours['hits']} hits here, {hits} in the paper")
        clusters.append(
            (
                hits,
                round(ours["player_accuracy"] * hits),
                round(ours["team_accuracy"] * hits),
                round(paper_player / 100 * hits),
                round(paper_team / 100 * hits),
            )
        )

    def metrics(sample: list[tuple[int, int, int, int, int]]) -> dict[str, float]:
        hits = sum(c[0] for c in sample)
        player, team = sum(c[1] for c in sample) / hits, sum(c[2] for c in sample) / hits
        return {
            "player": player,
            "team": team,
            "player_minus_paper": player - sum(c[3] for c in sample) / hits,
            "team_minus_paper": team - sum(c[4] for c in sample) / hits,
        }

    return bootstrap(clusters, metrics)


def shot_type() -> dict[str, list[float]]:
    """Table 1, what kind of shot: per held-out tournament, weighted by its hits."""
    accuracy = json.loads((METRICS / "shot_type_per_tournament.json").read_text())
    windows, _ = build_dataset(RALLY_FEATURES, SHOT_TYPE_LABELS)
    hits = collections.Counter(w.tournament for w in windows)
    clusters = [(hits[t], round(accuracy[t] * hits[t])) for t in sorted(accuracy)]

    def metrics(sample: list[tuple[int, int]]) -> dict[str, float]:
        return {"accuracy": sum(c[1] for c in sample) / sum(c[0] for c in sample)}

    return bootstrap(clusters, metrics)


def stages(rows: list[Row]) -> dict[str, float]:
    """The chain step by step: of the real hits of the four classes, the share that
    gets through a step and every step before it (detected, then the right player).
    Who hit is only annotated in VIGO, so the player steps count VIGO hits alone."""
    four_class = [r for r in rows if r.truth_s is not None and r.truth_type in CLASSES]
    vigo = [r for r in four_class if r.truth_player is not None]
    vigo_detected = [r for r in vigo if r.detected_s is not None]
    return {
        "detected": ratio(sum(r.detected_s is not None for r in four_class), len(four_class)),
        "vigo_detected": ratio(len(vigo_detected), len(vigo)),
        "vigo_detected_player": ratio(
            sum(r.player == r.truth_player for r in vigo_detected), len(vigo)
        ),
    }


def chain_rows() -> dict[str, list[Row]]:
    """Every real hit and every detection of chain_cv_hits.csv, by rally."""

    def number(text: str) -> float | None:
        return float(text) if text else None

    by_rally: dict[str, list[Row]] = collections.defaultdict(list)
    with (METRICS / "chain_cv_hits.csv").open() as handle:
        for r in csv.DictReader(handle, delimiter=";"):
            by_rally[r["rally"]].append(
                Row(
                    rally=r["rally"],
                    tournament=r["tournament"],
                    truth_s=number(r["truth_s"]),
                    detected_s=number(r["detected_s"]),
                    truth_type=r["truth_type"] or None,
                    truth_player=int(r["truth_player"]) if r["truth_player"] else None,
                    player=int(r["player"]) if r["player"] else None,
                    shot_type=r["shot_type"] or None,
                )
            )
    return by_rally


def chain() -> dict[str, list[float]]:
    """Table 2, the whole chain: every rally, and VIGO for who hit."""
    by_rally = chain_rows()

    def metrics_of(keys: tuple[str, ...]) -> Callable[[list[list[Row]]], dict[str, float]]:
        def metrics(sample: list[list[Row]]) -> dict[str, float]:
            rows = [row for rally in sample for row in rally]
            values = {**summarize(rows), **stages(rows)}
            return {key: values[key] for key in keys}

        return metrics

    every = list(by_rally.values())
    vigo = [rows for rally, rows in by_rally.items() if "VIGO" in rally]
    return {
        **bootstrap(
            every,
            metrics_of(("detection_f1", "type_accuracy", "end_to_end_when_what", "detected")),
        ),
        **bootstrap(
            vigo,
            metrics_of(
                (
                    "player_accuracy",
                    "team_accuracy",
                    "end_to_end_vigo",
                    "vigo_detected",
                    "vigo_detected_player",
                )
            ),
        ),
    }


def chain_hits() -> dict[str, int]:
    """How many real hits each part of the chain table is measured on."""
    rows = [row for rally in chain_rows().values() for row in rally]
    four_class = [r for r in rows if r.truth_s is not None and r.truth_type in CLASSES]
    return {
        "four_class": len(four_class),
        "vigo": sum(r.truth_player is not None for r in four_class),
    }


def main() -> None:
    intervals = {
        "tabla1_asignacion": assignment(),
        "tabla1_tipo": shot_type(),
        "tabla2_cadena": chain(),
    }
    for table, rows in intervals.items():
        for key, (point, low, high) in rows.items():
            print(f"{table:18s} {key:22s} {point:.4f}  [{low:.4f}, {high:.4f}]")
    audio = json.loads((METRICS / "audio_threshold_seeds.json").read_text())["metrics"]
    result = ExperimentResult(
        name="confidence_intervals",
        summary=(
            "Intervalos de confianza del 95% de las cifras principales, por remuestreo de "
            "rallies (de torneos en el tipo de golpe de la tabla 1)."
        ),
        metrics={
            f"{table}.{key}.{bound}": value
            for table, rows in intervals.items()
            for key, values in rows.items()
            for bound, value in zip(("valor", "bajo", "alto"), values, strict=True)
        },
        dataset="Los ficheros de evidencia de docs/metrics y la caché de etiquetas",
        method=f"Bootstrap por grupos, {RESAMPLES} remuestreos, percentiles 2,5 y 97,5",
        params={
            "resamples": RESAMPLES,
            "seed": SEED,
            "intervals": intervals,
            "chain_hits": chain_hits(),
        },
        notes=(
            "La unidad de remuestreo es el rally porque los golpes de un mismo rally no son "
            "independientes. La comparación con el paper es emparejada: los mismos 16 rallies "
            "(su tabla 3), remuestreados a la vez para los dos métodos. Si el intervalo de la "
            "diferencia incluye el 0, la diferencia no es concluyente. El detector de audio no "
            "tiene intervalo aquí: su cifra ya es la "
            f"media de {int(audio['runs'])} ejecuciones con su desviación típica "
            f"({audio['f1']:.3f} ± {audio['f1_std']:.3f})."
        ),
    )
    print(f"guardado -> {result.save()}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()

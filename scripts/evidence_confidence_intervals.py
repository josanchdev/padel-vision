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
from padel_ml.chain_eval import Row, summarize
from padel_ml.evidence import ExperimentResult
from padel_ml.shot_type_dataset import build_dataset

from padel_cv.paths import METRICS, RALLY_FEATURES, SHOT_TYPE_LABELS

RESAMPLES = 10_000
SEED = 0

PAPER_TABLE_3 = [
    (87.50, 87.50, 16), (83.87, 87.10, 31), (88.89, 88.89, 9), (61.54, 76.92, 13),
    (87.50, 87.50, 8), (100.00, 100.00, 10), (81.48, 81.48, 27), (88.24, 94.12, 17),
    (100.00, 100.00, 16), (77.14, 82.86, 35), (90.00, 100.00, 10), (86.96, 86.96, 46),
    (85.00, 90.00, 20), (80.95, 80.95, 21), (82.86, 85.71, 35), (40.00, 60.00, 5),
]  # fmt: skip
"""Decorte et al. (CVPRW 2024), Table 3: player %, team %, hits for each VIGO rally,
in the same order as ours (VIGO_00 ... VIGO_16, without VIGO_13)."""
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


def chain() -> dict[str, list[float]]:
    """Table 2, the whole chain: every rally, and VIGO for who hit."""

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

    def metrics_of(keys: tuple[str, ...]) -> Callable[[list[list[Row]]], dict[str, float]]:
        def metrics(sample: list[list[Row]]) -> dict[str, float]:
            summary = summarize([row for rally in sample for row in rally])
            return {key: summary[key] for key in keys}

        return metrics

    every = list(by_rally.values())
    vigo = [rows for rally, rows in by_rally.items() if "VIGO" in rally]
    return {
        **bootstrap(every, metrics_of(("detection_f1", "type_accuracy", "end_to_end_when_what"))),
        **bootstrap(vigo, metrics_of(("player_accuracy", "team_accuracy", "end_to_end_vigo"))),
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
        params={"resamples": RESAMPLES, "seed": SEED, "intervals": intervals},
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

"""The numbers the text prints with \\cifra{key}, each tied to its evidence."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from padel_ml.thesis.latex import (
    Number,
    count,
    number,
    percent,
    percent_range,
    points,
    proportion,
    proportion_range,
    quantity,
)
from padel_ml.thesis.sources import Evidence

Format = Callable[[Any], str]


class _File:
    """Numbers read from one evidence file, each one noting its own source."""

    def __init__(self, evidence: Evidence, name: str) -> None:
        self.data = evidence(name)
        self.name = name

    def __call__(self, key: str, format_: Format, *path: str) -> Number:
        value = self.data
        for part in path:
            value = value[part]
        return Number(key, format_(value), f"{self.name}.json: {'.'.join(path)}")


def _with_interval(
    evidence: Evidence, key: str, ci_key: str, value_format: Format, range_format: Any
) -> list[Number]:
    """key, key-ic, key-bajo and key-alto, from confidence_intervals.json."""
    value, low, high = evidence.interval(ci_key)
    source = f"confidence_intervals.json: {ci_key}"
    return [
        Number(key, value_format(value), source),
        Number(f"{key}-ic", range_format(low, high), source),
        Number(f"{key}-bajo", value_format(low), source),
        Number(f"{key}-alto", value_format(high), source),
    ]


def thesis_numbers(evidence: Evidence) -> list[Number]:
    """Every number of the text, in the order of the chapters."""
    audio = _File(evidence, "audio_threshold_seeds")
    same_data = _File(evidence, "method_comparison_same_data")
    assignment = _File(evidence, "hit_assignment_replica")
    classifier = _File(evidence, "shot_type_classifier")
    chain = _File(evidence, "chain_cv")
    dataset = _File(evidence, "dataset_shot_types")
    ball = _File(evidence, "ball_detector")
    cost = _File(evidence, "processing_time")

    def pct(key: str, ci_key: str) -> list[Number]:
        return _with_interval(evidence, key, ci_key, percent, percent_range)

    def prop(key: str, ci_key: str) -> list[Number]:
        return _with_interval(evidence, key, ci_key, proportion, proportion_range)

    def diff(key: str, ci_key: str) -> list[Number]:
        value, low, high = evidence.interval(ci_key)
        source = f"confidence_intervals.json: {ci_key}"
        return [
            Number(key, points(value), source),
            Number(f"{key}-bajo", points(low), source),
            Number(f"{key}-alto", points(high), source),
        ]

    one_decimal = partial(number, places=1)
    seconds = partial(quantity, places=1, unit=r"\second")
    pixels = partial(quantity, unit="px")
    from_percent = lambda value: percent(value / 100)  # noqa: E731
    cleaned = ("params", "by_tolerance", "detector + limpieza", "15px")
    serve = ("params", "class_weight_ablation", "0.0")
    funnel = ("params", "funnel_vigo_percent")

    return [
        # Cuándo: detector de audio, media de 3 ejecuciones (tabla 1).
        audio("audio-f1", proportion, "metrics", "f1"),
        audio("audio-f1-desviacion", proportion, "metrics", "f1_std"),
        audio("audio-precision", proportion, "metrics", "precision"),
        audio("audio-recall", proportion, "metrics", "recall"),
        audio("audio-ejecuciones", count, "metrics", "runs"),
        audio("audio-umbral", one_decimal, "params", "threshold"),
        audio("decorte-deteccion-f1", partial(proportion, places=2), "metrics", "paper_f1"),
        same_data("poseypelota-f1", proportion, "metrics", "old_f1"),
        same_data("audio-mismos-datos-f1", proportion, "metrics", "audio_f1"),
        # Quién: asignación con los instantes anotados (tabla 1).
        *pct("asignacion-jugador", "tabla1_asignacion.player"),
        *pct("asignacion-pareja", "tabla1_asignacion.team"),
        *diff("asignacion-diferencia-jugador", "tabla1_asignacion.player_minus_paper"),
        *diff("asignacion-diferencia-pareja", "tabla1_asignacion.team_minus_paper"),
        assignment("asignacion-golpes", count, "metrics", "hits"),
        assignment("asignacion-ventana", count, "params", "window_half_frames"),
        assignment("decorte-jugador", percent, "metrics", "paper_player_accuracy"),
        assignment("decorte-pareja", percent, "metrics", "paper_team_accuracy"),
        Number(
            "asignacion-puntos",
            count(len(evidence("hit_assignment_per_rally"))),
            "hit_assignment_per_rally.json: número de puntos",
        ),
        # Qué tipo: clasificador por torneos (tabla 1).
        *pct("tipo-accuracy", "tabla1_tipo.accuracy"),
        classifier("tipo-macro-f1", proportion, "metrics", "macro_f1"),
        classifier("tipo-golpes", count, "metrics", "n"),
        classifier("tipo-torneos", count, "metrics", "tournaments"),
        classifier("tipo-accuracy-sin-regla", percent, "metrics", "accuracy_without_serve_rule"),
        classifier("tipo-macro-f1-sin-regla", proportion, "metrics", "macro_f1_without_serve_rule"),
        classifier("saque-precision", proportion, *serve, "serve_precision"),
        classifier("saque-recall", proportion, *serve, "serve_recall"),
        classifier(
            "saque-precision-sin-regla", proportion, "params", "serve_without_rule", "precision"
        ),
        # El sistema completo por torneos (tabla 2).
        *prop("cadena-deteccion-f1", "tabla2_cadena.detection_f1"),
        chain("cadena-deteccion-precision", proportion, "metrics", "detection_precision"),
        chain("cadena-deteccion-recall", proportion, "metrics", "detection_recall"),
        *pct("cadena-tipo", "tabla2_cadena.type_accuracy"),
        chain("cadena-tipo-macro-f1", proportion, "metrics", "type_macro_f1"),
        *pct("cadena-jugador", "tabla2_cadena.player_accuracy"),
        *pct("cadena-pareja", "tabla2_cadena.team_accuracy"),
        *pct("cadena-detectado-tipo", "tabla2_cadena.end_to_end_when_what"),
        *pct("cadena-todo-vigo", "tabla2_cadena.end_to_end_vigo"),
        chain("cadena-golpes", count, "metrics", "real_hits"),
        chain("cadena-golpes-vigo", count, "metrics", "vigo_hits"),
        chain("embudo-detectados", from_percent, *funnel, "Detectados"),
        chain("embudo-jugador", from_percent, *funnel, "Jugador correcto"),
        chain("embudo-jugador-tipo", from_percent, *funnel, "Jugador y tipo correctos"),
        # El dataset propio.
        dataset("dataset-golpes", count, "metrics", "labelled"),
        dataset("dataset-entrenables", count, "metrics", "trainable"),
        dataset("dataset-descartes", count, "metrics", "discards"),
        dataset("dataset-descartes-pct", percent, "metrics", "discard_rate"),
        dataset("dataset-puntos", count, "metrics", "rallies"),
        dataset("dataset-torneos", count, "metrics", "tournaments"),
        dataset("dataset-desbalance", one_decimal, "metrics", "imbalance_ratio"),
        dataset("dataset-golpes-por-punto", count, "metrics", "median_hits_per_rally"),
        dataset("dataset-derecha", count, "per_class", "Forehand", "support"),
        dataset("dataset-reves", count, "per_class", "Backhand", "support"),
        dataset("dataset-remate", count, "per_class", "Smash", "support"),
        dataset("dataset-saque", count, "per_class", "Serve", "support"),
        # La pelota, en un partido que el modelo no vio.
        ball("pelota-f1", proportion, "metrics", "f1"),
        ball("pelota-precision", proportion, "metrics", "precision"),
        ball("pelota-recall", proportion, "metrics", "recall"),
        ball("pelota-tolerancia", partial(pixels, places=0), "metrics", "tolerance_px"),
        ball("pelota-error-mediano", partial(pixels, places=1), "metrics", "median_error_px"),
        ball("pelota-limpieza-f1", proportion, *cleaned, "f1"),
        ball("pelota-limpieza-recall", proportion, *cleaned, "recall"),
        # Coste de procesado.
        cost("coste-total", seconds, "metrics", "total_s"),
        cost("coste-video", seconds, "metrics", "video_s"),
        cost("coste-factor", partial(number, places=2), "metrics", "realtime_factor"),
    ]

"""Evidence for the whole chain on CVSPORTS: numbers, per-hit table and figures.

Reads what `scripts/evaluate_chain_cv.py` left under runs/chain_cv/ (the
per-hit rows and each fold's training curve) and writes:

    docs/metrics/chain_cv.json                    every metric, per sync mode
    docs/metrics/chain_cv_hits.csv                every hit, the table behind them
    docs/metrics/figures/chain_cv_funnel.png      from real hit to fully right
    docs/metrics/figures/chain_cv_per_tournament.png
    docs/metrics/figures/chain_cv_type_confusion.png
    docs/metrics/figures/shot_classifier_training_curve.png

    uv run python scripts/evidence_chain_cv.py
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in WSL with no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from padel_ml.chain_eval import MODES, Row, summarize
from padel_ml.evidence import (
    FIGURES_DIR,
    METRICS_DIR,
    ExperimentResult,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    plot_confusion,
)
from padel_ml.shot_type_dataset import CLASSES

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "chain_cv"

# Reference palette (dataviz skill, light mode): categorical slots 1-2, ink, grid.
SERIES_1 = "#2a78d6"  # synchronised (the system as it ships)
SERIES_2 = "#eb6834"  # not synchronised (the audio instant as heard)
MUTED_LINE = "#b9b8b2"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
SPANISH = {"Forehand": "Derecha", "Backhand": "Revés", "Smash": "Remate", "Serve": "Saque"}


def _style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_2, labelsize=9)
    axes.yaxis.label.set_color(INK_2)
    axes.xaxis.label.set_color(INK_2)


def _funnel(rows: list[Row], out: Path) -> dict[str, dict[str, float]]:
    """VIGO, the only tournament where every step has ground truth."""
    vigo = [
        r
        for r in rows
        if r.truth_s is not None and r.truth_player is not None and r.truth_type in CLASSES
    ]
    stages = ["Golpes reales", "Detectados", "Jugador correcto", "Jugador y tipo correctos"]
    values: dict[str, dict[str, float]] = {}
    for mode in ("none", "video"):
        detected = [r for r in vigo if r.detected_s is not None]
        player = [r for r in detected if r.player[mode] == r.truth_player]
        both = [r for r in player if r.shot_type[mode] == r.truth_type]
        values[mode] = {
            stage: 100 * len(group) / len(vigo)
            for stage, group in zip(stages, [vigo, detected, player, both], strict=True)
        }

    figure, axes = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    x = np.arange(len(stages))
    width = 0.38
    for offset, mode, colour, name in (
        (-width / 2, "none", SERIES_2, "Instante del audio, sin sincronizar"),
        (width / 2, "video", SERIES_1, "Sincronizado por vídeo (sistema final)"),
    ):
        heights = [values[mode][s] for s in stages]
        bars = axes.bar(x + offset, heights, width - 0.04, color=colour, label=name, zorder=2)
        for bar, height in zip(bars, heights, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.5,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    axes.set_xticks(x, stages)
    axes.set_ylim(0, 112)
    axes.set_ylabel("% de los golpes reales")
    axes.yaxis.grid(True, color=GRID, zorder=0)
    _style(axes)
    axes.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    axes.set_title(
        f"De cada 100 golpes reales (VIGO, {len(vigo)} golpes, torneo no visto al entrenar)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return values


def _per_tournament(rows: list[Row], out: Path) -> dict[str, dict[str, float]]:
    tournaments = sorted({r.tournament for r in rows})
    numbers = {t: summarize([r for r in rows if r.tournament == t], "video") for t in tournaments}
    overall = summarize(rows, "video")["end_to_end_when_what"]
    ordered = sorted(tournaments, key=lambda t: numbers[t]["end_to_end_when_what"])
    values = [100 * numbers[t]["end_to_end_when_what"] for t in ordered]

    figure, axes = plt.subplots(figsize=(7.0, 4.2), facecolor=SURFACE)
    y = np.arange(len(ordered))
    axes.barh(y, values, 0.62, color=SERIES_1, zorder=2)
    for yi, value in zip(y, values, strict=True):
        axes.text(value + 0.8, float(yi), f"{value:.0f}%", va="center", fontsize=8, color=INK)
    axes.axvline(100 * overall, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=3)
    axes.text(
        100 * overall, len(ordered) - 0.35, f" global {100 * overall:.0f}%", fontsize=8, color=INK_2
    )
    axes.set_yticks(y, [t.split("_", 1)[1].title() for t in ordered])
    axes.set_xlim(0, 100)
    axes.set_xlabel("Golpes detectados y con el tipo correcto (%)")
    axes.xaxis.grid(True, color=GRID, zorder=0)
    _style(axes)
    axes.set_title(
        "Cadena completa por torneo, cada uno sin verse al entrenar",
        fontsize=10,
        color=INK,
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return numbers


def _training_curves(out: Path) -> dict[str, float]:
    curves = []
    for fold in sorted(p for p in RUNS.iterdir() if (p / "bst.pt").exists()):
        checkpoint = torch.load(fold / "bst.pt", map_location="cpu", weights_only=False)
        curves.append(checkpoint["curve"])
    loss = np.array([c["loss"] for c in curves])
    accuracy = 100 * np.array([c["held_out_accuracy"] for c in curves])
    epochs = np.arange(1, loss.shape[1] + 1)

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.4), facecolor=SURFACE)
    for axes, data, label, title in (
        (left, loss, "Pérdida de entrenamiento", "Pérdida (entrenamiento)"),
        (right, accuracy, "Accuracy (%)", "Accuracy en el torneo reservado"),
    ):
        for series in data:
            axes.plot(epochs, series, color=MUTED_LINE, linewidth=0.8, zorder=1)
        axes.plot(epochs, data.mean(axis=0), color=SERIES_1, linewidth=2, zorder=2)
        axes.set_xlabel("Época")
        axes.set_ylabel(label)
        axes.grid(True, color=GRID, zorder=0)
        _style(axes)
        axes.set_title(title, fontsize=10, color=INK, loc="left")
    right.text(
        epochs[-1],
        accuracy.mean(axis=0)[-1] + 1.5,
        f"media {accuracy.mean(axis=0)[-1]:.1f}%",
        ha="right",
        fontsize=8,
        color=INK,
    )
    figure.suptitle(
        f"Clasificador de tipo: {len(curves)} rondas (gris) y su media (azul)",
        fontsize=10,
        color=INK_2,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    final = accuracy[:, -1]
    return {
        "folds": float(len(curves)),
        "held_out_accuracy_final_mean": round(float(final.mean()), 2),
        "held_out_accuracy_best_epoch_mean": round(float(accuracy.max(axis=1).mean()), 2),
        "loss_first_epoch_mean": round(float(loss[:, 0].mean()), 4),
        "loss_last_epoch_mean": round(float(loss[:, -1].mean()), 4),
    }


def main() -> None:
    rows = [Row(**r) for r in json.loads((RUNS / "all_rows.json").read_text())]

    metrics: dict[str, float] = {}
    for mode in MODES:
        metrics |= {f"{mode}/{k}": v for k, v in summarize(rows, mode).items()}

    typed = [
        r
        for r in rows
        if r.truth_s is not None and r.detected_s is not None and r.truth_type in CLASSES
    ]
    labels = [*CLASSES, "Unclassified"]
    matrix = confusion_matrix(
        [r.truth_type or "" for r in typed], [r.shot_type["video"] for r in typed], labels
    )
    per_class = per_class_metrics(matrix, labels)
    per_class.pop("Unclassified", None)
    metrics["video/type_macro_f1"] = macro_f1(per_class)

    funnel = _funnel(rows, FIGURES_DIR / "chain_cv_funnel.png")
    tournaments = _per_tournament(rows, FIGURES_DIR / "chain_cv_per_tournament.png")
    curves = _training_curves(FIGURES_DIR / "shot_classifier_training_curve.png")
    plot_confusion(
        matrix,
        [SPANISH.get(c, c) for c in CLASSES] + ["Sin clasificar"],
        f"Tipo de golpe, cadena completa ({len(typed)} golpes)",
        FIGURES_DIR / "chain_cv_type_confusion.png",
    )

    lags = {}
    for fold in sorted(p for p in RUNS.iterdir() if (p / "lags.json").exists()):
        data = json.loads((fold / "lags.json").read_text())
        per_rally = [v["video_lag_s"] for v in data["per_rally"].values()]
        lags[fold.name] = {
            "global_lag_ms": round(1000 * data["global_lag_s"], 1),
            "video_lag_ms_median": round(1000 * statistics.median(per_rally), 1),
        }

    video = summarize(rows, "video")
    result = ExperimentResult(
        name="chain_cv",
        summary=(
            f"Cadena completa en CVSPORTS, dejando fuera cada torneo: deteccion F1 "
            f"{video['detection_f1']:.3f}, tipo {video['type_accuracy']:.1%}, jugador (VIGO) "
            f"{video['player_accuracy']:.1%}; detectado y bien clasificado "
            f"{video['end_to_end_when_what']:.1%}, todo correcto en VIGO "
            f"{video['end_to_end_vigo']:.1%}."
        ),
        metrics=metrics,
        dataset="CVSPORTS: 99 rallies / 11 torneos (cuando, que); VIGO con jugador anotado (quien)",
        method=(
            "leave-one-tournament-out: audio CRNN + BST-0 reentrenados por ronda; "
            "pose y pelota cacheadas"
        ),
        params={
            "modes": list(MODES),
            "collar_s": 0.25,
            "funnel_vigo_percent": funnel,
            "per_tournament_video": tournaments,
            "training_curve": curves,
            "audio_lag_per_fold": lags,
        },
        per_class=per_class,
        confusion=matrix.tolist(),
        labels=labels,
        notes=(
            "Primera medida de la cadena entera con los instantes que detecta el audio. Hasta "
            "aqui cada paso se media aislado: el quien con instantes anotados (89,65%) y el "
            "tipo con el golpeador ya elegido (81,84%). 'none' usa el instante del audio tal "
            "cual; 'global' el desfase medio de los torneos de entrenamiento; 'video' el "
            "desfase estimado sin etiquetas en cada video (el sistema final)."
        ),
    )
    path = result.save()
    with (METRICS_DIR / "chain_cv_hits.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "rally",
                "truth_s",
                "detected_s",
                "truth_type",
                "truth_player",
                *[f"player_{m}" for m in MODES],
                *[f"type_{m}" for m in MODES],
            ]
        )
        for r in rows:
            record = asdict(r)
            writer.writerow(
                [
                    r.rally,
                    r.truth_s,
                    r.detected_s,
                    r.truth_type,
                    r.truth_player,
                    *[record["player"][m] for m in MODES],
                    *[record["shot_type"][m] for m in MODES],
                ]
            )
    for key, value in video.items():
        print(f"  {key:30s} {value}")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()

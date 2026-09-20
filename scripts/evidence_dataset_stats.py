"""Dataset statistics for the shot-type labels (ADR-0016) into docs/metrics/.

Our own labelling effort is a contribution of the dissertation, so its numbers
belong in the evidence folder like any measurement: class balance, per-tournament
split, discard rate and the imbalance the training has to compensate for.

    uv run python scripts/evidence_dataset_stats.py
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from padel_ml.evidence import FIGURES_DIR, ExperimentResult

from padel_cv.shot_type_annotator import load_marks

LABELS_DIR = Path("data/labels/types")
TRAINABLE = ["Forehand", "Backhand", "Smash", "Serve"]


def main() -> None:
    per_tournament: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    per_rally: list[int] = []
    discards_per_rally: list[int] = []
    total: collections.Counter[str] = collections.Counter()

    for path in sorted(LABELS_DIR.glob("*.csv")):
        marks = load_marks(path)
        tournament = "_".join(path.stem.split("_")[:2])
        counts = collections.Counter(m.shot_type for m in marks.values() if m.shot_type)
        per_tournament[tournament].update(counts)
        total.update(counts)
        per_rally.append(sum(counts.values()))
        discards_per_rally.append(counts.get("Other", 0))

    labelled = sum(total.values())
    trainable = labelled - total["Other"]
    largest = max(total[c] for c in TRAINABLE)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [*TRAINABLE, "Other"]
    spanish = {
        "Forehand": "Derecha",
        "Backhand": "Revés",
        "Smash": "Remate",
        "Serve": "Saque",
        "Other": "Descarte",
    }
    colours = ["#3b7dd8", "#d88c3b", "#3bd87d", "#d84a3b", "#999999"]
    values = [total[n] for n in names]
    bars = axes[0].bar([spanish[n] for n in names], values, color=colours)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value}\n{value / labelled:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title(
        f"Etiquetado propio: {labelled} golpes\n"
        f"({trainable} entrenables + {total['Other']} descartes)"
    )
    axes[0].set_ylabel("golpes")
    axes[0].set_ylim(0, max(values) * 1.25)

    tournaments = sorted(per_tournament)
    bottom = [0.0] * len(tournaments)
    for name, colour in zip(names, colours, strict=True):
        heights = [per_tournament[t][name] for t in tournaments]
        axes[1].bar(
            [t.split("_")[1] for t in tournaments],
            heights,
            bottom=bottom,
            label=spanish[name],
            color=colour,
        )
        bottom = [b + h for b, h in zip(bottom, heights, strict=True)]
    axes[1].set_title("Reparto por torneo (permite split cross-torneo)")
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "dataset_shot_types.png", dpi=150)
    plt.close(figure)

    result = ExperimentResult(
        name="dataset_shot_types",
        summary=(
            f"Etiquetado propio del tipo de golpe: {labelled} golpes en "
            f"{len(per_rally)} rallies de {len(per_tournament)} torneos "
            f"({trainable} entrenables, {total['Other']} descartes)."
        ),
        metrics={
            "labelled": labelled,
            "trainable": trainable,
            "discards": total["Other"],
            "discard_rate": round(total["Other"] / labelled, 4),
            "rallies": len(per_rally),
            "tournaments": len(per_tournament),
            "imbalance_ratio": round(largest / min(total[c] for c in TRAINABLE), 2),
            "median_hits_per_rally": statistics.median(per_rally),
        },
        dataset="CVSPORTS_Padel — instantes del GT publicado, tipo etiquetado por nosotros",
        method="Anotador propio: clip en bucle a velocidad real, una tecla por golpe",
        params={
            "classes": TRAINABLE,
            "clip_half_frames": 25,
            "class_weights_suggested": {c: round(largest / total[c], 2) for c in TRAINABLE},
        },
        per_class={c: {"support": total[c], "share": round(total[c] / labelled, 4)} for c in names},
        notes=(
            "Los descartes (Other) NO se entrenan: sirven para calibrar el umbral de rechazo "
            "(ADR-0016 D). El saque está desbalanceado 8,4:1 por construcción del dataset (un "
            "saque por punto), no por sesgo del etiquetado, y habrá que pesarlo en la pérdida. "
            "La distribución estable entre los 11 torneos habilita un split cross-torneo, más "
            "exigente que cross-rally porque mide generalización a pistas y cámaras nuevas."
        ),
    )
    path = result.save()
    (Path("docs/metrics") / "dataset_per_tournament.json").write_text(
        json.dumps({t: dict(c) for t, c in sorted(per_tournament.items())}, indent=2) + "\n"
    )
    print(f"{labelled} golpes · {trainable} entrenables · {len(per_tournament)} torneos")
    print(f"guardado -> {path}")


if __name__ == "__main__":
    main()

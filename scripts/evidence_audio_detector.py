"""Regenerate the audio hit-detector evidence (ADR-0015 A) into docs/metrics/.

This is the experiment that justifies the whole change of direction, so it must
be reproducible, not remembered: it trains the CRNN on CVSPORTS_Padel with a
cross-rally split, evaluates event-based (250 ms collar) and saves metrics, the
threshold sweep and the comparison against every earlier approach we tried.

One run is not enough to choose a threshold: the run-to-run spread (~0.02 F1) is
as large as the gap between neighbouring thresholds. So it trains once per seed
— each seed draws its own 70/30 split of the rallies and its own initial
weights — and averages them (`audio_threshold_seeds.json`, the cited F1). The
single-run files come from seed 0.

    uv run python scripts/evidence_audio_detector.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from padel_ml.audio_detector import AudioHitCRNN
from padel_ml.audio_train import (
    DEFAULT_THRESHOLD,
    build_all_rallies,
    event_eval,
    fit,
    peaks_from_frames,
)
from padel_ml.evidence import FIGURES_DIR, ExperimentResult, plot_comparison

from padel_cv.cvsports import load_hits_csv

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
CACHE = REPO / "data" / "datasets" / "audio_cache.npz"
PAPER_F1 = 0.92

#: What we measured BEFORE the audio pivot, so the report can show why it changed.
#: Sources: docs/experiments.md (each of these was measured on our own footage).
PRIOR_APPROACHES = {
    "Heurística muñeca-pelota (ADR-0012)": {"recall": 0.52, "note": "15-52% según vídeo"},
    "Detector aprendido por ventana": {
        "recall": 0.29,
        "note": "0,60 en test balanceado, 29% recall al deslizar",
    },
    "Localizer por frame (pose+pelota)": {
        "recall": 0.61,
        "note": "61% recall / 70% prec, solo en rallies",
    },
}


def sweep_thresholds(
    model: AudioHitCRNN,
    val_rallies: list,
    hits: dict,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> list[dict[str, float]]:
    """F1/precision/recall across thresholds — shows 0,5 is a choice, not luck."""
    rows: list[dict[str, float]] = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = fp = fn = 0
        for rally in val_rallies:
            feats = ((rally.features - mean[0]) / std[0]).astype(np.float32)
            with torch.no_grad():
                probs = (
                    torch.sigmoid(model(torch.from_numpy(feats)[None].to(device)))[0].cpu().numpy()
                )
            peaks = peaks_from_frames(probs, threshold=threshold, min_gap_frames=8)
            a, b, c = event_eval(peaks, hits.get(rally.filename, []))
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        print(f"  thr {threshold:.1f}: F1 {f1:.3f}  P {precision:.3f}  R {recall:.3f}")
    return rows


@dataclass
class SeedRun:
    """One training run: its split, its loss curve and its threshold sweep."""

    seed: int
    n_rallies: int
    n_train: int
    n_val: int
    losses: list[float]
    sweep: list[dict[str, float]]


def train_and_sweep(seed: int) -> SeedRun:
    """Train on a seeded 70/30 split of the rallies and sweep the threshold.

    Seeded so repeated runs are comparable: the run-to-run spread (~0.02 F1) is
    as large as the gap between neighbouring thresholds, so an unseeded sweep
    cannot tell a real difference from luck. Not about bit-exact reproduction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rallies = build_all_rallies(DATASET, CACHE)
    rng = np.random.default_rng(seed)
    index = np.arange(len(rallies))
    rng.shuffle(index)
    n_val = int(len(rallies) * 0.3)
    val_rallies = [rallies[i] for i in index[:n_val]]
    train_rallies = [rallies[i] for i in index[n_val:]]

    print(f"entrenando en {device} ({len(train_rallies)} rallies train / {n_val} val)")
    # The same training function as the production model and every chain fold.
    fitted = fit(train_rallies, epochs=30, seed=seed, device=device)
    hits = load_hits_csv(DATASET / "metadata" / "hits.csv")
    print("barrido de umbral:")
    sweep = sweep_thresholds(fitted.model, val_rallies, hits, fitted.mean, fitted.std, device)
    return SeedRun(seed, len(rallies), len(train_rallies), n_val, fitted.losses, sweep)


def report_single_run(run: SeedRun) -> None:
    """The seed-0 files: metrics, loss curve, threshold sweep and the comparison."""
    sweep, losses, seed = run.sweep, run.losses, run.seed
    best = max(sweep, key=lambda r: r["f1"])
    chosen = next(r for r in sweep if r["threshold"] == DEFAULT_THRESHOLD)

    # loss curve + threshold sweep figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, len(losses) + 1), losses, color="#3b7dd8")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("focal BCE loss")
    axes[0].set_title("Entrenamiento del CRNN de audio")
    axes[0].grid(alpha=0.3)
    thresholds = [r["threshold"] for r in sweep]
    for key, color in [("f1", "#3b7dd8"), ("precision", "#d84a3b"), ("recall", "#3bd87d")]:
        axes[1].plot(thresholds, [r[key] for r in sweep], marker="o", label=key, color=color)
    axes[1].axvline(DEFAULT_THRESHOLD, linestyle="--", color="grey", linewidth=1)
    axes[1].set_xlabel("umbral")
    axes[1].set_title("Sensibilidad al umbral (eval event-based, collar 250 ms)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "audio_detector_training.png", dpi=150)
    plt.close(figure)

    comparison = {name: {"recall": value["recall"]} for name, value in PRIOR_APPROACHES.items()}
    comparison["AUDIO CRNN (elegido)"] = {"recall": chosen["recall"]}
    plot_comparison(
        comparison,
        "recall",
        "Por qué el audio: recall de cada enfoque probado",
        FIGURES_DIR / "shot_detection_approaches.png",
    )

    result = ExperimentResult(
        name="audio_hit_detector",
        summary=(
            f"Detector de golpes por audio (CRNN SED): F1 {chosen['f1']:.3f}, "
            f"precisión {chosen['precision']:.3f}, recall {chosen['recall']:.3f} "
            f"(paper {PAPER_F1})."
        ),
        metrics={
            "f1": chosen["f1"],
            "precision": chosen["precision"],
            "recall": chosen["recall"],
            "tp": chosen["tp"],
            "fp": chosen["fp"],
            "fn": chosen["fn"],
            "paper_f1": PAPER_F1,
            "best_f1_any_threshold": best["f1"],
        },
        dataset=(
            f"CVSPORTS_Padel — {run.n_rallies} rallies, 2.377 golpes; "
            f"split cross-rally {run.n_train}/{run.n_val}"
        ),
        method="CRNN log-Mel (3xconv2D pool-frecuencia + 2xGRU bi) + focal BCE",
        params={
            "sample_rate": 48000,
            "n_mels": 40,
            "n_fft": 2048,
            "hop": 1024,
            "seq_len": 256,
            "epochs": 30,
            "batch_size": 32,
            "lr": 1e-3,
            "threshold": DEFAULT_THRESHOLD,
            "seed": seed,
            "min_gap_frames": 8,
            "collar_s": 0.25,
        },
        notes=(
            "Reproduce el paper de pádel (Decorte CVPRW 2024, F1 0,92) sobre su dataset "
            "abierto. Justifica el giro: los enfoques previos por pose+pelota daban "
            "recall 15-52% (heurística), 29% (detector por ventana deslizado sobre vídeo "
            "real) y 61% (localizer por frame, solo en rallies). El recall que se pierde "
            "son dejadas/slices, que suenan poco — el mismo modo de fallo que reporta el "
            "paper (su tasa de borrado 0,13). Bug corregido en el camino: las stats de "
            "normalización deben calcularse sobre features CRUDAS y reutilizarse en "
            "evaluación; calcularlas después de normalizar hundía la precisión a 0,53."
        ),
    )
    path = result.save()
    Path("docs/metrics/audio_threshold_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
    Path("docs/metrics/audio_training_loss.json").write_text(json.dumps(losses, indent=2) + "\n")
    print(f"\nguardado -> {path}")


def report_seeds(runs: list[SeedRun]) -> None:
    """Average every seed per threshold: the evidence the threshold is chosen on."""
    thresholds = [r["threshold"] for r in runs[0].sweep]
    table: dict[str, dict[str, float]] = {}
    for i, threshold in enumerate(thresholds):
        rows = [run.sweep[i] for run in runs]
        table[f"{threshold:.1f}"] = {
            "f1_mean": round(statistics.mean(r["f1"] for r in rows), 4),
            "f1_std": round(statistics.stdev(r["f1"] for r in rows), 4),
            "precision_mean": round(statistics.mean(r["precision"] for r in rows), 4),
            "recall_mean": round(statistics.mean(r["recall"] for r in rows), 4),
        }
    chosen = table[f"{DEFAULT_THRESHOLD:.1f}"]
    best = max(table, key=lambda t: table[t]["f1_mean"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [t for t in table if 0.3 <= float(t) <= 0.7]
    x = [float(t) for t in shown]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        x, [table[t]["precision_mean"] for t in shown], "s--", color="#d84a3b", label="precisión"
    )
    axis.plot(x, [table[t]["recall_mean"] for t in shown], "^--", color="#3bd87d", label="recall")
    axis.axvline(
        DEFAULT_THRESHOLD,
        linestyle=":",
        color="grey",
        label=f"elegido: {DEFAULT_THRESHOLD:.1f}".replace(".", ","),
    )
    axis.errorbar(
        x,
        [table[t]["f1_mean"] for t in shown],
        yerr=[table[t]["f1_std"] for t in shown],
        fmt="o-",
        color="#3b7dd8",
        capsize=4,
        label=f"F1 (media de {len(runs)} ejecuciones)",
    )
    axis.set_xlabel("umbral de detección")
    axis.set_ylabel("métrica")
    axis.set_title(
        f"Elección del umbral: media de {len(runs)} ejecuciones\n"
        "(la barra de error es la dispersión entre ejecuciones)"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "audio_threshold_seeds.png", dpi=150)
    plt.close(figure)

    result = ExperimentResult(
        name="audio_threshold_seeds",
        summary=(
            f"Detector de golpes por audio, media de {len(runs)} ejecuciones: F1 "
            f"{chosen['f1_mean']:.3f} ± {chosen['f1_std']:.3f} con umbral "
            f"{DEFAULT_THRESHOLD} (precisión {chosen['precision_mean']:.3f}, recall "
            f"{chosen['recall_mean']:.3f}; paper {PAPER_F1})."
        ),
        metrics={
            "f1": chosen["f1_mean"],
            "f1_std": chosen["f1_std"],
            "precision": chosen["precision_mean"],
            "recall": chosen["recall_mean"],
            "runs": len(runs),
            "paper_f1": PAPER_F1,
        },
        dataset=(
            f"CVSPORTS_Padel — {runs[0].n_rallies} rallies; cada ejecución con su propio "
            f"split cross-rally {runs[0].n_train}/{runs[0].n_val}"
        ),
        method="CRNN log-Mel + focal BCE; evaluación event-based con collar de 250 ms",
        params={
            "seeds": [run.seed for run in runs],
            "threshold": DEFAULT_THRESHOLD,
            "by_threshold": table,
        },
        notes=(
            f"Mejor umbral por F1 medio: {best}; el sistema usa {DEFAULT_THRESHOLD}. Cada "
            "semilla cambia el reparto de rallies entre entrenamiento y validación y los "
            "pesos iniciales, así que la media equivale a promediar particiones, como hace "
            "el paper con 4. Las ejecuciones individuales no son idénticas bit a bit en GPU."
        ),
    )
    print(f"guardado -> {result.save()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=3, help="Training runs to average")
    args = parser.parse_args()
    runs = []
    for seed in range(args.seeds):
        print(f"\n=== semilla {seed} ===")
        runs.append(train_and_sweep(seed))
    report_single_run(runs[0])
    if len(runs) > 1:  # a spread needs at least two runs
        report_seeds(runs)


if __name__ == "__main__":
    main()

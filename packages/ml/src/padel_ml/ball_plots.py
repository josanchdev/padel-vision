"""Presentation plots for the ball-detector comparison (ADR-0009, Decision E).

Self-contained PNGs meant to drop straight into defense slides: training curves,
and the V2-vs-V3 bar chart broken down by visible vs occluded frames — the plot
that shows *where* the modern model earns its keep. No interactivity, no external
assets; just matplotlib figures saved to disk (and logged to MLflow upstream).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the WSL box / CI
import matplotlib.pyplot as plt

from padel_ml.ball_metrics import BallEval


def plot_training_curves(history: dict[str, list[float]], out_path: Path, title: str) -> None:
    """F1/precision/recall per epoch for one training run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(next(iter(history.values()))) + 1)
    for name, values in history.items():
        ax.plot(epochs, values, label=name, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_occlusion_comparison(
    results: dict[str, BallEval], out_path: Path, title: str = "F1 por visibilidad de la pelota"
) -> None:
    """Grouped bars: each model's F1 on visible vs occluded vs overall frames.

    results maps model name -> its BallEval. This is the headline comparison:
    if TrackNetV3 wins, the occluded bar is where the gap shows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(results)
    groups = ["overall", "visible", "occluded"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        ev = results[model]
        scores = [ev.overall.f1, ev.visible.f1, ev.occluded.f1]
        positions = [g + i * width for g in range(len(groups))]
        bars = ax.bar(positions, scores, width=width, label=model)
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.01,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks([g + width * (len(models) - 1) / 2 for g in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

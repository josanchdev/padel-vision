"""Persist experiment evidence for the dissertation (see docs/metrics/README.md).

Every measurement that backs a claim in the report must survive as DATA, not
just as prose in the log: a number written in a paragraph cannot be re-plotted,
re-checked by a reader, or defended if asked "where does this come from?".

So each experiment writes a JSON file under `docs/metrics/` (versioned in git,
unlike `runs/`) with its metrics, its confusion matrix and the context needed to
reproduce it — plus optional figures under `docs/metrics/figures/`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

METRICS_DIR = Path("docs/metrics")
FIGURES_DIR = METRICS_DIR / "figures"


def _git_commit() -> str | None:
    """Commit the measurement was taken at, so a result can be traced to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class ExperimentResult:
    """One measured experiment, with everything needed to cite and rerun it."""

    name: str
    """Short id, e.g. "audio_detector_crnn"."""
    summary: str
    """One line: what was measured and what came out."""
    metrics: dict[str, float]
    """Headline numbers (f1, precision, accuracy...)."""
    dataset: str = ""
    """What it was measured ON — the split matters as much as the number."""
    method: str = ""
    """Which approach produced it, for method-vs-method tables."""
    params: dict[str, Any] = field(default_factory=dict)
    """Hyper-parameters/config, so the run can be repeated."""
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-class precision/recall/f1 — a macro average hides the rare classes."""
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    notes: str = ""
    """Why this experiment was run and what was concluded (the "why" of the ADR)."""
    commit: str | None = field(default_factory=_git_commit)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def save(self, directory: Path = METRICS_DIR) -> Path:
        """Write to `<directory>/<name>.json` and return the path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")
        return path


def load_result(name: str, directory: Path = METRICS_DIR) -> ExperimentResult:
    """Read one saved experiment back."""
    data = json.loads((directory / f"{name}.json").read_text())
    return ExperimentResult(**data)


def load_all(directory: Path = METRICS_DIR) -> list[ExperimentResult]:
    """Every saved experiment, newest first."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(ExperimentResult(**json.loads(path.read_text())))
        except TypeError:  # older files with a different shape
            continue
    return sorted(out, key=lambda r: r.timestamp, reverse=True)


def confusion_matrix(
    true_labels: list[str], predicted: list[str | None], labels: list[str]
) -> npt.NDArray[np.int64]:
    """Counts of true (rows) vs predicted (columns).

    `None` predictions land in a trailing "unassigned"/"rejected" column when
    `labels` includes one, so abstentions stay visible instead of being dropped.
    """
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(true_labels, predicted, strict=True):
        if truth not in index:
            continue
        key = prediction if prediction in index else labels[-1]
        matrix[index[truth], index[key]] += 1
    return matrix


def per_class_metrics(
    matrix: npt.NDArray[np.int64], labels: list[str]
) -> dict[str, dict[str, float]]:
    """Precision, recall and F1 for each class, from a confusion matrix."""
    out: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        true_positive = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())
        actual = int(matrix[i, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual,
        }
    return out


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Unweighted mean F1 — the metric that exposes weak rare classes."""
    scores = [m["f1"] for m in per_class.values() if m["support"] > 0]
    return round(float(np.mean(scores)), 4) if scores else 0.0


def plot_confusion(
    matrix: npt.NDArray[np.int64],
    labels: list[str],
    title: str,
    out_path: Path,
    normalize: bool = True,
) -> Path:
    """Save a confusion-matrix figure (normalized by row = recall per class)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs in WSL with no display
    import matplotlib.pyplot as plt

    data = matrix.astype(float)
    if normalize:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)

    figure, axes = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 2))
    image = axes.imshow(data, cmap="Blues", vmin=0, vmax=data.max() or 1)
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("Predicho")
    axes.set_ylabel("Real")
    axes.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(matrix[i, j])
            text = f"{data[i, j]:.2f}\n({count})" if normalize else str(count)
            axes.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] > data.max() * 0.6 else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_comparison(
    results: dict[str, dict[str, float]],
    metric: str,
    title: str,
    out_path: Path,
    reference: tuple[str, float] | None = None,
) -> Path:
    """Bar chart comparing one metric across methods.

    `reference` draws a horizontal line (e.g. the paper's published score), which
    is what turns "our number" into "our number vs the state of the art".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    values = [results[n].get(metric, 0.0) for n in names]
    figure, axes = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    bars = axes.bar(names, values, color="#3b7dd8")
    if reference is not None:
        label, value = reference
        axes.axhline(value, color="#d84a3b", linestyle="--", linewidth=1.5, label=label)
        axes.legend()
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes.set_ylabel(metric)
    axes.set_title(title)
    axes.set_ylim(0, max(max(values, default=1.0), 1.0) * 1.15)
    axes.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path

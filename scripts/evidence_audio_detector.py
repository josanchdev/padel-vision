"""Regenerate the audio hit-detector evidence (ADR-0015 A) into docs/metrics/.

This is the experiment that justifies the whole change of direction, so it must
be reproducible, not remembered: it trains the CRNN on CVSPORTS_Padel with a
cross-rally split, evaluates event-based (250 ms collar) and saves metrics, the
threshold sweep and the comparison against every earlier approach we tried.

    uv run python scripts/evidence_audio_detector.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from padel_ml.audio_dataset import load_hits_csv
from padel_ml.audio_detector import AudioHitCRNN, focal_bce_loss
from padel_ml.audio_train import (
    DEFAULT_THRESHOLD,
    _sequences,
    build_all_rallies,
    event_eval,
    peaks_from_frames,
)
from padel_ml.evidence import FIGURES_DIR, ExperimentResult, plot_comparison

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


def main(seed: int = 0) -> None:
    # Seeded so repeated runs are comparable: run-to-run spread (~0.02 F1) is as
    # large as the gap between neighbouring thresholds, so an unseeded sweep
    # cannot tell a real difference from luck. Not about bit-exact reproduction.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rallies = build_all_rallies(DATASET, CACHE)
    rng = np.random.default_rng(seed)
    index = np.arange(len(rallies))
    rng.shuffle(index)
    n_val = int(len(rallies) * 0.3)
    val_rallies = [rallies[i] for i in index[:n_val]]
    train_rallies = [rallies[i] for i in index[n_val:]]

    xtr, ytr = _sequences(train_rallies)
    mean = xtr.mean(axis=(0, 1), keepdims=True)
    std = xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    xtr = ((xtr - mean) / std).astype(np.float32)

    model = AudioHitCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=32,
        shuffle=True,
    )
    losses: list[float] = []
    print(f"entrenando en {device} ({len(train_rallies)} rallies train / {n_val} val)")
    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_bce_loss(model(xb.to(device)), yb.to(device))
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss)
        losses.append(round(epoch_loss / len(loader), 5))
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: loss {losses[-1]:.4f}")

    model.eval()
    hits = load_hits_csv(DATASET / "metadata" / "hits.csv")
    print("barrido de umbral:")
    sweep = sweep_thresholds(model, val_rallies, hits, mean, std, device)
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
            f"CVSPORTS_Padel — {len(rallies)} rallies, 2.377 golpes; "
            f"split cross-rally {len(train_rallies)}/{n_val}"
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


if __name__ == "__main__":
    main()

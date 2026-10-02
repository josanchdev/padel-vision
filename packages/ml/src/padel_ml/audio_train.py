"""Train + evaluate the audio hit detector (ADR-0015, replica of Decorte 2024).

Builds log-Mel features for every rally, splits cross-rally (whole rallies to one
side, as the paper does), trains the CRNN with focal loss on fixed-length
sequences, and evaluates with EVENT-BASED metrics (a predicted hit peak counts if
it falls within a collar of a real hit — the paper uses 250 ms). The headline is
F1; the paper reports 92%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from padel_cv.arrays import FloatArray
from padel_cv.cvsports import load_hits_csv
from padel_ml.audio_dataset import SEQ_LEN, RallyAudio, build_rally, frame_time
from padel_ml.audio_detector import AudioHitCRNN, focal_bce_loss

DEFAULT_THRESHOLD = 0.5
"""Detection threshold, measured rather than assumed.

Swept over 0.3-0.7, mean of three runs (docs/metrics/audio_threshold_seeds.json):
F1 0.932 / 0.947 / **0.957** / 0.953 / 0.942. 0.5 wins on F1 and balances
precision (0.969) against recall (0.945) without sacrificing either, and the curve
is flat around it, so the choice is robust rather than a fragile peak. Dropping
to 0.4 buys recall (0.958) at the cost of precision (0.938) if soft hits — drop
shots, slices — ever matter more than false positives.
"""


def build_all_rallies(dataset_dir: Path, cache: Path | None = None) -> list[RallyAudio]:
    """Build (or load cached) log-Mel features + labels for every rally mp4."""
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        return [
            RallyAudio(f.astype(np.float32), lab.astype(np.float32), str(n))
            for f, lab, n in zip(data["feats"], data["labels"], data["names"], strict=True)
        ]
    hits = load_hits_csv(dataset_dir / "metadata" / "hits.csv")
    rallies: list[RallyAudio] = []
    for mp4 in sorted((dataset_dir / "rallies").glob("*.mp4")):
        rallies.append(build_rally(mp4, hits.get(mp4.name, [])))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            feats=np.array([r.features for r in rallies], dtype=object),
            labels=np.array([r.labels for r in rallies], dtype=object),
            names=np.array([r.filename for r in rallies]),
        )
    return rallies


def _sequences(rallies: list[RallyAudio], seq_len: int = SEQ_LEN) -> tuple[FloatArray, FloatArray]:
    """Cut each rally into non-overlapping seq_len windows (pad the last)."""
    xs, ys = [], []
    for r in rallies:
        n = len(r.features)
        for start in range(0, max(n, 1), seq_len):
            fx = r.features[start : start + seq_len]
            fy = r.labels[start : start + seq_len]
            if len(fx) < seq_len:  # pad the tail
                pad = seq_len - len(fx)
                fx = np.pad(fx, ((0, pad), (0, 0)))
                fy = np.pad(fy, (0, pad))
            xs.append(fx)
            ys.append(fy)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def peaks_from_frames(
    probs: FloatArray, threshold: float = DEFAULT_THRESHOLD, min_gap_frames: int = 4
) -> list[int]:
    """Frame indices of local maxima above threshold, min_gap apart (one per hit)."""
    out: list[int] = []
    last = -(10**9)
    for i in range(1, len(probs) - 1):
        if probs[i] < threshold:
            continue
        if probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1] and i - last >= min_gap_frames:
            out.append(i)
            last = i
    return out


def match_events(
    predicted_s: list[float], truth_s: list[float], collar_s: float = 0.25
) -> list[tuple[int, int]]:
    """Pair predicted events with real ones: (predicted index, truth index).

    Each prediction, in order, takes the nearest still-unmatched real event
    within `collar_s` (the paper's collar is 250 ms). One-to-one: a real hit can
    be claimed once, so two detections of the same hit count one TP and one FP.
    """
    matched: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for p, pt in enumerate(predicted_s):
        best_i, best_d = -1, collar_s
        for i, tt in enumerate(truth_s):
            if i in matched:
                continue
            d = abs(pt - tt)
            if d <= best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            matched.add(best_i)
            pairs.append((p, best_i))
    return pairs


def event_eval(
    pred_frames: list[int], true_windows: list[tuple[float, float]], collar_s: float = 0.25
) -> tuple[int, int, int]:
    """Event-based TP/FP/FN: a predicted peak matches a real hit if its time is
    within `collar_s` of the hit window (paper's collar = 250 ms)."""
    true_times = [(s + e) / 2 for s, e in true_windows]
    tp = len(match_events([frame_time(pf) for pf in pred_frames], true_times, collar_s))
    return tp, len(pred_frames) - tp, len(true_times) - tp


@dataclass
class FittedDetector:
    """A trained CRNN plus the standardisation it was trained with."""

    model: AudioHitCRNN
    mean: FloatArray
    std: FloatArray
    losses: list[float]
    """Mean training loss per epoch."""


def fit(
    rallies: list[RallyAudio], epochs: int = 30, seed: int = 0, device: str | None = None
) -> FittedDetector:
    """The one training loop of the audio detector.

    Production, every evaluation fold and the evidence all train through here,
    so the model that is measured is trained exactly like the one that runs.
    `seed` fixes the initial weights and the batch order (GPU kernels may still
    differ in the last digits).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    xtr, ytr = _sequences(rallies)
    # Standardisation stats from the RAW training features, kept to apply to
    # whatever is scored later: computed after normalising they come out ~0/~1,
    # mis-scale the evaluation and sink the precision.
    mean = xtr.mean(axis=(0, 1), keepdims=True)
    std = xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    xtr = ((xtr - mean) / std).astype(np.float32)

    model = AudioHitCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=32,
        shuffle=True,
    )
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_bce_loss(model(xb.to(device)), yb.to(device))
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss)
        losses.append(round(epoch_loss / len(loader), 5))
    model.eval()
    return FittedDetector(model, mean.astype(np.float32), std.astype(np.float32), losses)


def fit_and_save(
    dataset_dir: Path,
    out_path: Path,
    cache: Path | None = None,
    exclude: set[str] | None = None,
    epochs: int = 30,
    seed: int = 0,
    device: str | None = None,
) -> Path:
    """Train on every rally (minus `exclude`) and save weights + norm stats.

    Nothing is held back, so the saved detector is as strong as the data allows;
    the evaluations score it on rallies they excluded themselves.
    """
    rallies = build_all_rallies(dataset_dir, cache)
    excluded = exclude or set()
    fitted = fit([r for r in rallies if r.filename not in excluded], epochs, seed, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": fitted.model.state_dict(), "mean": fitted.mean, "std": fitted.std},
        out_path,
    )
    return out_path


def hit_probabilities(
    audio_source: Path, checkpoint: Path, device: str | None = None
) -> FloatArray:
    """Per-spectrogram-frame hit probability for a video/audio file."""
    from padel_ml.audio_dataset import build_rally

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = AudioHitCRNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]
    feats = build_rally(audio_source, []).features
    feats = ((feats - mean[0]) / std[0]).astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(feats)[None].to(device)))[0].cpu().numpy()
    return probs.astype(np.float32)


def detect_hits_in_audio(
    audio_source: Path,
    checkpoint: Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_frames: int = 8,
    device: str | None = None,
) -> list[float]:
    """Run the saved detector over a video/audio file → list of hit times (s)."""
    probs = hit_probabilities(audio_source, checkpoint, device)
    return [frame_time(p) for p in peaks_from_frames(probs, threshold, min_gap_frames)]

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
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

from padel_ml.audio_dataset import (
    SEQ_LEN,
    RallyAudio,
    build_rally,
    frame_time,
    load_hits_csv,
)
from padel_ml.audio_detector import AudioHitCRNN, focal_bce_loss

FloatArray = npt.NDArray[np.float32]

DEFAULT_THRESHOLD = 0.5
"""Detection threshold, measured rather than assumed.

Swept over 0.3-0.7 (docs/metrics/audio_threshold_seeds.json): mean F1 0.930 /
0.947 / **0.956** / 0.954 / 0.943. 0.5 wins on F1 and balances precision (0.969)
against recall (0.944) without sacrificing either, and the curve is flat around
it, so the choice is robust rather than a fragile peak. Dropping to 0.4 buys
recall (0.960) at the cost of precision (0.936) if soft hits — drop shots,
slices — ever matter more than false positives.
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


def windows_from_frames(
    probs: FloatArray, threshold: float = DEFAULT_THRESHOLD, min_gap_frames: int = 4
) -> list[tuple[int, int]]:
    """Contiguous above-threshold runs as (start, end) spectrogram frames.

    The paper drives hit assignment from the predicted hit WINDOW (padded to
    500 ms only when it is shorter), not from a bare peak — so we emit the real
    onset/offset the CRNN produces. The window's width is itself information: a
    slice sounds different from a smash. Runs closer than `min_gap_frames` are
    merged, as they belong to one hit.
    """
    above = probs >= threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, hot in enumerate(above):
        if hot and start is None:
            start = i
        elif not hot and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(probs) - 1))
    merged: list[tuple[int, int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_gap_frames:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    return merged


def hit_windows_seconds(
    probs: FloatArray, threshold: float = DEFAULT_THRESHOLD, min_gap_frames: int = 4
) -> list[tuple[float, float]]:
    """Predicted hit windows as (start, end) seconds."""
    return [
        (frame_time(a), frame_time(b))
        for a, b in windows_from_frames(probs, threshold, min_gap_frames)
    ]


@dataclass
class AudioEval:
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def event_eval(
    pred_frames: list[int], true_windows: list[tuple[float, float]], collar_s: float = 0.25
) -> tuple[int, int, int]:
    """Event-based TP/FP/FN: a predicted peak matches a real hit if its time is
    within `collar_s` of the hit window (paper's collar = 250 ms)."""
    true_times = [(s + e) / 2 for s, e in true_windows]
    matched: set[int] = set()
    tp = 0
    for pf in pred_frames:
        pt = frame_time(pf)
        best_i, best_d = -1, collar_s
        for i, tt in enumerate(true_times):
            if i in matched:
                continue
            d = abs(pt - tt)
            if d <= best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            matched.add(best_i)
            tp += 1
    fp = len(pred_frames) - tp
    fn = len(true_times) - len(matched)
    return tp, fp, fn


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

    Unlike `train_audio_detector` (which holds a val split back to report F1),
    this trains on all available data so the saved detector is as strong as
    possible, and persists the standardization stats needed at inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rallies = build_all_rallies(dataset_dir, cache)
    exclude = exclude or set()
    train_r = [r for r in rallies if r.filename not in exclude]
    xtr, ytr = _sequences(train_r)
    mean = xtr.mean(axis=(0, 1), keepdims=True)
    std = xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    xtr = ((xtr - mean) / std).astype(np.float32)

    model = AudioHitCRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=32,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_bce_loss(model(xb.to(device)), yb.to(device))
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
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


def detect_hit_windows_in_audio(
    audio_source: Path,
    checkpoint: Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_frames: int = 8,
    device: str | None = None,
) -> list[tuple[float, float]]:
    """Hit WINDOWS (start, end) in seconds — what hit assignment consumes."""
    probs = hit_probabilities(audio_source, checkpoint, device)
    return hit_windows_seconds(probs, threshold, min_gap_frames)


def train_audio_detector(
    dataset_dir: Path,
    cache: Path | None = None,
    val_frac: float = 0.3,
    epochs: int = 30,
    seed: int = 0,
    device: str | None = None,
) -> AudioEval:
    """Train on 70% of rallies, evaluate event-based on the held-out 30%."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rallies = build_all_rallies(dataset_dir, cache)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rallies))
    rng.shuffle(idx)
    n_val = int(len(rallies) * val_frac)
    val_ids, train_ids = set(idx[:n_val].tolist()), set(idx[n_val:].tolist())
    train_r = [rallies[i] for i in train_ids]
    val_r = [rallies[i] for i in val_ids]

    xtr, ytr = _sequences(train_r)
    # Standardization stats from RAW train features — keep them to apply to the
    # eval rallies too (computing them after normalizing would give ~0/~1 and
    # mis-scale the eval, tanking precision).
    mean = xtr.mean(axis=(0, 1), keepdims=True)
    std = xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    xtr = ((xtr - mean) / std).astype(np.float32)

    model = AudioHitCRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=32,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_bce_loss(model(xb.to(device)), yb.to(device))
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

    # evaluate event-based per rally on the held-out set (normalize with train stats)
    model.eval()
    hits = load_hits_csv(dataset_dir / "metadata" / "hits.csv")
    tp = fp = fn = 0
    for r in val_r:
        feats = ((r.features - mean[0]) / std[0]).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(feats)[None].to(device)))[0].cpu().numpy()
        peaks = peaks_from_frames(probs, threshold=DEFAULT_THRESHOLD, min_gap_frames=8)
        t, f, n = event_eval(peaks, hits.get(r.filename, []))
        tp, fp, fn = tp + t, fp + f, fn + n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AudioEval(f1=f1, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)

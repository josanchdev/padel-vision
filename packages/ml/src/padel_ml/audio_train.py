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


def _normalize(train_x: FloatArray, *others: FloatArray) -> list[FloatArray]:
    """Standardize log-Mel by train mean/std (per feature)."""
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    return [((a - mean) / std).astype(np.float32) for a in (train_x, *others)]


def peaks_from_frames(
    probs: FloatArray, threshold: float = 0.5, min_gap_frames: int = 4
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
    xva, _yva = _sequences(val_r)
    xtr, xva = _normalize(xtr, xva)

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

    # evaluate event-based per rally on the held-out set
    model.eval()
    hits = load_hits_csv(dataset_dir / "metadata" / "hits.csv")
    mean = xtr.mean(axis=(0, 1), keepdims=True)
    std = xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    tp = fp = fn = 0
    for r in val_r:
        feats = ((r.features - mean[0]) / std[0]).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(feats)[None].to(device)))[0].cpu().numpy()
        peaks = peaks_from_frames(probs)
        t, f, n = event_eval(peaks, hits.get(r.filename, []))
        tp, fp, fn = tp + t, fp + f, fn + n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AudioEval(f1=f1, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)

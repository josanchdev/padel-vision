"""Train the per-frame shot localizer (ShotLocalizer) and slide it over a video.

The window detector saturates in rallies (a plateau over a burst of shots). The
localizer emits a logit PER FRAME, trained on dense per-frame labels, so each
impact becomes a sharp peak. This module builds the dense dataset, trains, and
provides sliding inference that turns a video's pose+ball into one probability
per frame, from which shot events are the peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

from padel_ml.shot_detect_dataset import _HALF, _window_around
from padel_ml.shot_detector import ShotLocalizer

FloatArray = npt.NDArray[np.float32]


def train_localizer(
    npz_path: Path,
    val_match: int,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    d_model: int = 64,
    out_path: Path | None = None,
) -> ShotLocalizer:
    """Train on all matches except `val_match`. Dense per-frame BCE, the rare
    impact frames upweighted so the model doesn't predict all-zero."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(npz_path)
    pose = torch.from_numpy(data["pose"].astype(np.float32))
    ball = torch.from_numpy(data["ball"].astype(np.float32))
    y = torch.from_numpy(data["labels"].astype(np.float32))  # (N, T)
    matches = data["matches"]
    tr = matches != val_match

    pos = float(y[tr].sum())
    neg = float(y[tr].numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ShotLocalizer(d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(pose[tr], ball[tr], y[tr]), batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for p, b, yy in loader:
            opt.zero_grad()
            loss_fn(model(p.to(device), b.to(device)), yy.to(device)).backward()
            opt.step()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "d_model": d_model}, out_path)
    return model


@dataclass
class ShotEventLoc:
    """A localized shot: the frame of the probability peak and its height."""

    frame: int
    prob: float


def sliding_probs(
    model: ShotLocalizer,
    persons_by_frame: dict[int, list[FloatArray]],
    ball_by_frame: dict[int, tuple[float, float]],
    lo: int,
    hi: int,
    device: str | None = None,
) -> dict[int, float]:
    """One shot-probability per frame in [lo, hi), from the centre-frame logit of
    the window centred there. This is the sliding evaluation the video sees."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out: dict[int, float] = {}
    for c in range(lo, hi):
        w = _window_around(c, persons_by_frame, ball_by_frame)
        if w is None:
            continue
        with torch.no_grad():
            pose = torch.from_numpy(w[0])[None].to(device)
            ball = torch.from_numpy(w[1])[None].to(device)
            logits = model(pose, ball)[0]  # (T,)
        out[c] = float(torch.sigmoid(logits[_HALF]))
    return out


def sliding_probs_averaged(
    model: ShotLocalizer,
    persons_by_frame: dict[int, list[FloatArray]],
    ball_by_frame: dict[int, tuple[float, float]],
    lo: int,
    hi: int,
    step: int = 4,
    device: str | None = None,
) -> dict[int, float]:
    """Per-frame probability AVERAGED over every overlapping window that predicts
    it. The localizer predicts all T frames of a window, so a frame gets several
    votes from windows centred nearby; averaging them cancels per-window noise and
    sharpens the peaks (a real impact is predicted consistently, noise isn't).
    `step` strides the window centres (4 = 8x fewer forward passes than every-frame).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    from collections import defaultdict

    acc: dict[int, list[float]] = defaultdict(list)
    for c in range(lo, hi, step):
        w = _window_around(c, persons_by_frame, ball_by_frame)
        if w is None:
            continue
        with torch.no_grad():
            pose = torch.from_numpy(w[0])[None].to(device)
            ball = torch.from_numpy(w[1])[None].to(device)
            probs = torch.sigmoid(model(pose, ball)[0]).cpu().numpy()  # (T,)
        for i, f in enumerate(range(c - _HALF, c - _HALF + len(probs))):
            if lo <= f < hi:
                acc[f].append(float(probs[i]))
    return {f: float(np.mean(v)) for f, v in acc.items()}


def peaks_from_probs(
    probs: dict[int, float],
    threshold: float = 0.5,
    min_gap: int = 8,
    min_run: int = 1,
    run_window: int = 5,
) -> list[ShotEventLoc]:
    """Local maxima of the per-frame signal above `threshold`, at least `min_gap`
    frames apart — one event per shot (replaces the failed window-level NMS).

    `min_run`/`run_window`: a real shot gives a SUSTAINED response, a noise spike
    doesn't. A peak only counts if at least `min_run` of the `run_window` frames
    centred on it are above `threshold` (e.g. 4 of 5). This drops the isolated
    "pum, shot" false peaks Jorge saw while keeping true shots.
    """
    events: list[ShotEventLoc] = []
    half = run_window // 2
    for f in sorted(probs):
        p = probs[f]
        if p < threshold:
            continue
        lo = probs.get(f - 1, 0.0)
        hi = probs.get(f + 1, 0.0)
        if not (p >= lo and p >= hi):  # local max only
            continue
        if min_run > 1:  # require a sustained run around the peak
            window = range(f - half, f - half + run_window)
            run = sum(1 for k in window if probs.get(k, 0.0) >= threshold)
            if run < min_run:
                continue
        if events and f - events[-1].frame < min_gap:
            if p > events[-1].prob:  # keep the stronger of two close peaks
                events[-1] = ShotEventLoc(f, p)
        else:
            events.append(ShotEventLoc(f, p))
    return events

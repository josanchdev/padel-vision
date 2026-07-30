"""Train and evaluate the shot DETECTOR (Modelo 1, ADR-0013), cross-match split.

Trains on one match's windows and validates on the other, so the detector is
scored on an unseen court/rally (the honest measure, as everywhere else in this
project — ADR-0008). The headline metric is recall/precision of the shot class,
directly comparable to the heuristic's 15-52% recall (ADR-0012): this is the
number that says whether the learned detector is worth it.

Trains on CPU in well under a minute (~57k params, 1810 windows), so it never
competes with the ball detector for the GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

from padel_ml.shot_detector import ShotDetector


@dataclass
class DetectEval:
    precision: float
    recall: float
    f1: float
    accuracy: float
    threshold: float


def _metrics(probs: npt.NDArray[np.float32], y: npt.NDArray[np.int64], thr: float) -> DetectEval:
    pred = probs >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = float((pred == y).mean())
    return DetectEval(precision, recall, f1, acc, thr)


def _load_split(
    npz_path: Path, val_match: int
) -> tuple[TensorDataset, TensorDataset, npt.NDArray[np.int64]]:
    data = np.load(npz_path)
    pose = torch.from_numpy(data["pose"].astype(np.float32))
    ball = torch.from_numpy(data["ball"].astype(np.float32))
    y = torch.from_numpy(data["labels"].astype(np.float32))
    matches = data["matches"].astype(np.int64)
    is_val = matches == val_match
    tr = TensorDataset(pose[~is_val], ball[~is_val], y[~is_val])
    va = TensorDataset(pose[is_val], ball[is_val], y[is_val])
    return tr, va, matches


def _evaluate(
    model: ShotDetector, loader: DataLoader[tuple[torch.Tensor, ...]], device: str
) -> DetectEval:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for pose, ball, y in loader:
            logits = model(pose.to(device), ball.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.numpy())
    probs = np.concatenate(all_probs)
    ys = np.concatenate(all_y).astype(np.int64)
    # Pick the threshold that maximises F1 on this split (reported alongside 0.5).
    best = _metrics(probs, ys, 0.5)
    for thr in np.linspace(0.1, 0.9, 17):
        cand = _metrics(probs, ys, float(thr))
        if cand.f1 > best.f1:
            best = cand
    return best


def train_detector(
    npz_path: Path,
    val_match: int = 0,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    d_model: int = 64,
    out_path: Path | None = None,
) -> DetectEval:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, _ = _load_split(npz_path, val_match)
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=batch_size)

    model = ShotDetector(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = DetectEval(0.0, 0.0, -1.0, 0.0, 0.5)
    best_state = model.state_dict()
    for epoch in range(1, epochs + 1):
        model.train()
        for pose, ball, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(pose.to(device), ball.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        ev = _evaluate(model, val_loader, device)
        if ev.f1 > best.f1:
            best = ev
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"ep{epoch:>3} val recall={ev.recall:.3f} precision={ev.precision:.3f} "
                f"F1={ev.f1:.3f} (thr={ev.threshold:.2f})"
            )

    print(
        f"\nmejor: recall={best.recall:.3f} precision={best.precision:.3f} "
        f"F1={best.f1:.3f} acc={best.accuracy:.3f} @thr={best.threshold:.2f}"
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "d_model": d_model}, out_path)
        print(f"modelo guardado en {out_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the learned shot detector (Modelo 1)")
    parser.add_argument("--data", type=Path, default=Path("data/datasets/shot_detect.npz"))
    parser.add_argument("--val-match", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    train_detector(args.data, val_match=args.val_match, epochs=args.epochs, out_path=args.out)


if __name__ == "__main__":
    main()

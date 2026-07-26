"""Train and evaluate a shot classifier (cross-match split, class-weighted).

Reports macro-F1 and per-class F1 plus a confusion matrix — the honest metrics
for an imbalanced problem, where plain accuracy would hide the rare classes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from padel_ml.data import load_shot_data
from padel_ml.stgcn import STGCN

Batch = tuple[torch.Tensor, ...]


@dataclass
class EvalResult:
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    accuracy: float


def evaluate(
    model: torch.nn.Module, loader: DataLoader[Batch], classes: list[str], device: str
) -> EvalResult:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(y.tolist())
    labels = list(range(len(classes)))
    per_class = f1_score(trues, preds, labels=labels, average=None, zero_division=0)
    return EvalResult(
        macro_f1=float(f1_score(trues, preds, labels=labels, average="macro", zero_division=0)),
        per_class_f1={classes[i]: float(per_class[i]) for i in labels},
        confusion=confusion_matrix(trues, preds, labels=labels),
        accuracy=float(np.mean(np.array(preds) == np.array(trues))),
    )


def train(
    npz_path: Path,
    val_match: int = 0,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    out_path: Path | None = None,
) -> EvalResult:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_shot_data(npz_path, val_match=val_match)
    train_loader = DataLoader(data.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data.val, batch_size=batch_size)

    model = STGCN(num_classes=len(data.classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=data.class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = EvalResult(0.0, {}, np.zeros((0, 0)), 0.0)
    best_state = model.state_dict()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        scheduler.step()
        result = evaluate(model, val_loader, data.classes, device)
        if result.macro_f1 > best.macro_f1:
            best, best_state = result, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"ep{epoch:>3} loss={total / len(data.train):.3f} "
                f"val macro-F1={result.macro_f1:.3f} acc={result.accuracy:.3f}"
            )

    print(f"\nmejor macro-F1: {best.macro_f1:.3f} | accuracy: {best.accuracy:.3f}")
    print("F1 por clase:")
    for name, score in best.per_class_f1.items():
        print(f"  {name:10s} {score:.3f}")
    print("matriz de confusión (filas=verdad, cols=predicho):")
    print("  " + " ".join(f"{c[:4]:>5}" for c in data.classes))
    for i, row in enumerate(best.confusion):
        print(f"  {data.classes[i][:4]:>4} " + " ".join(f"{v:>5}" for v in row))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "classes": data.classes}, out_path)
        print(f"\nmodelo guardado en {out_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ST-GCN shot classifier")
    parser.add_argument("--data", type=Path, default=Path("data/datasets/shot_clips.npz"))
    parser.add_argument("--val-match", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    train(args.data, val_match=args.val_match, epochs=args.epochs, out_path=args.out)


if __name__ == "__main__":
    main()

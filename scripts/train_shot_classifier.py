"""Train the production shot-type classifier on ALL the data (ADR-0016).

Cross-validation (`cross_tournament_cv`) measures how well the approach
generalises by holding tournaments out in turn, but every model it trains is
thrown away. This trains the one that ships: all 11 tournaments, all 2,209
labelled hits, nothing held back — the standard practice of validating to learn
the score, then fitting on everything to deploy.

    uv run python scripts/train_shot_classifier.py [--epochs 60]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, build_dataset
from padel_ml.shot_type_model import ShotTypeBST
from padel_ml.shot_type_train import class_weights, to_tensors

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "datasets" / "rally_features"
LABELS = REPO / "data" / "labels" / "types"
OUT = REPO / "runs" / "shot_type" / "bst0.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    windows, stats = build_dataset(FEATURES, LABELS)
    print(f"{len(windows)} ventanas de {stats['rallies']} rallies · dispositivo {device}")

    pose, ball, labels = to_tensors(windows)
    weights = class_weights(labels, len(CLASSES))
    print("pesos de clase:", {c: round(float(w), 2) for c, w in zip(CLASSES, weights, strict=True)})

    torch.manual_seed(0)
    model = ShotTypeBST(seq_len=SEQ_LEN, n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pose, ball, labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    warmup = 8
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: (e + 1) / warmup
        if e < warmup
        else 0.5 * (1 + torch.cos(torch.tensor(torch.pi * (e - warmup) / (args.epochs - warmup)))),
    )

    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for pose_b, ball_b, y_b in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(pose_b.to(device), ball_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            mean_loss = total / len(loader)
            print(f"  epoca {epoch + 1:3d}/{args.epochs}  loss {mean_loss:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seq_len": SEQ_LEN,
            "classes": CLASSES,
            "n_train": len(windows),
            "epochs": args.epochs,
        },
        args.out,
    )
    print(f"\nentrenado con {len(windows)} golpes en {time.perf_counter() - started:.0f}s")
    print(f"guardado -> {args.out}")


if __name__ == "__main__":
    main()

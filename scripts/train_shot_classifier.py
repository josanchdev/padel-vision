"""Train the production shot-type classifier on ALL the data (ADR-0016).

Cross-validation (`cross_tournament_cv`) measures how well the approach
generalises by holding tournaments out in turn, but every model it trains is
thrown away. This trains the one that ships: all 11 tournaments, every labelled
hit, nothing held back — the standard practice of validating to learn the score,
then fitting on everything to deploy.

    uv run python scripts/train_shot_classifier.py [--epochs 60]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, build_dataset
from padel_ml.shot_type_train import EPOCHS, fit

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "datasets" / "rally_features"
LABELS = REPO / "data" / "labels" / "types"
OUT = REPO / "runs" / "shot_type" / "bst0.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    windows, stats = build_dataset(FEATURES, LABELS)
    print(f"{len(windows)} ventanas de {stats['rallies']} rallies · dispositivo {device}")

    started = time.perf_counter()
    # The same training function as every cross-validation fold: the model that
    # ships is trained exactly like the models that were measured.
    model, curve = fit(windows, SEQ_LEN, epochs=args.epochs, device=device)
    for epoch in range(9, len(curve.loss), 10):
        print(f"  epoca {epoch + 1:3d}/{args.epochs}  loss {curve.loss[epoch]:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seq_len": SEQ_LEN,
            "classes": CLASSES,
            "n_train": len(windows),
            "epochs": args.epochs,
            "loss_curve": curve.loss,
        },
        args.out,
    )
    print(f"\nentrenado con {len(windows)} golpes en {time.perf_counter() - started:.0f}s")
    print(f"guardado -> {args.out}")


if __name__ == "__main__":
    main()

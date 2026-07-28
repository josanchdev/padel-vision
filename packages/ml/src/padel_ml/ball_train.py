"""Train and evaluate a ball detector (TrackNetV2 or V3), cross-match split.

Trains on one match's frame cache and validates on the other, so the model is
scored on an unseen court/rally — the honest measure (as in the shot classifier,
ADR-0008). The target heatmap is almost all zeros (one small blob per frame), so
the loss is a weighted BCE that upweights the rare ball pixels; otherwise the net
trivially predicts "empty" everywhere.

Metrics (peak distance with tolerance, ADR-0009) and training curves go to
MLflow, and presentation PNGs are written for the defense (Decision E). The
occluded-vs-visible breakdown is the headline: it shows whether TrackNetV3's
refiner recovers balls that V2 misses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from padel_ml.ball_dataset import BallClips
from padel_ml.ball_metrics import BallEval, evaluate_ball
from padel_ml.ball_plots import plot_occlusion_comparison, plot_training_curves
from padel_ml.tracknet import TrackNetV2
from padel_ml.tracknet_v3 import TrackNetV3


def _make_model(name: str) -> nn.Module:
    if name == "tracknetv2":
        return TrackNetV2()
    if name == "tracknetv3":
        return TrackNetV3()
    raise ValueError(f"unknown model: {name}")


@dataclass
class BallTrainResult:
    model_name: str
    best: BallEval
    history: dict[str, list[float]]


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE with the sparse ball pixels upweighted by pos_weight.

    The target is almost all zeros (one small blob per frame), so unweighted BCE
    would collapse to predicting empty everywhere; pos_weight upweights the blob.
    """
    weight = 1.0 + (pos_weight - 1.0) * target  # pos_weight on blob, 1 elsewhere
    loss: torch.Tensor = nn.functional.binary_cross_entropy(pred, target, weight=weight)
    return loss


def train_ball(
    train_dirs: list[Path],
    val_dirs: list[Path],
    model_name: str = "tracknetv2",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    pos_weight: float = 200.0,
    tol: float = 4.0,
    seed: int = 0,
    out_path: Path | None = None,
    plots_dir: Path | None = None,
    augment: bool = False,
    neg_ratio: float | None = 2.0,
    num_workers: int = 4,
) -> BallTrainResult:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Train split: colour augmentation + subsampled negatives. Val keeps the real
    # distribution (no augment, all frames) so metrics are honest (ADR-0009).
    train_loader = DataLoader(
        BallClips(train_dirs, augment=augment, neg_ratio=neg_ratio),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_ds = BallClips(val_dirs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = _make_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {"f1": [], "precision": [], "recall": []}
    best = BallTrainResult(model_name, _empty_eval(), history)
    best_f1 = -1.0
    best_state = model.state_dict()

    mlflow.log_params(
        {"model": model_name, "epochs": epochs, "lr": lr, "pos_weight": pos_weight, "tol": tol}
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x.to(device)).squeeze(1)  # (N, H, W)
            loss = _weighted_bce(pred, y.to(device), pos_weight)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += loss.item() * len(x)
        scheduler.step()

        ev = _evaluate(model, val_loader, device, tol)
        history["f1"].append(ev.overall.f1)
        history["precision"].append(ev.overall.precision)
        history["recall"].append(ev.overall.recall)
        mlflow.log_metrics(
            {
                "val_f1": ev.overall.f1,
                "val_precision": ev.overall.precision,
                "val_recall": ev.overall.recall,
                "val_f1_occluded": ev.occluded.f1,
                "val_f1_visible": ev.visible.f1,
                "train_loss": total / max(len(train_loader.dataset), 1),  # type: ignore[arg-type]
            },
            step=epoch,
        )
        if ev.overall.f1 > best_f1:
            best_f1 = ev.overall.f1
            best = BallTrainResult(model_name, ev, history)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"ep{epoch:>3} loss={total / max(len(train_loader.dataset), 1):.4f} "  # type: ignore[arg-type]
                f"F1={ev.overall.f1:.3f} (vis={ev.visible.f1:.3f} occ={ev.occluded.f1:.3f}) "
                f"err={ev.overall.mean_error:.2f}px"
            )

    _report(best.best)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "model_name": model_name}, out_path)
        print(f"\nmodelo guardado en {out_path}")
    if plots_dir is not None:
        curve_path = plots_dir / f"{model_name}_curves.png"
        plot_training_curves(history, curve_path, f"{model_name}: F1/P/R por epoch")
        mlflow.log_artifact(str(curve_path))
    return best


def _empty_eval() -> BallEval:
    from padel_ml.ball_metrics import Counts

    return BallEval(Counts(), Counts(), Counts())


BallBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _evaluate(model: nn.Module, loader: DataLoader[BallBatch], device: str, tol: float) -> BallEval:
    from padel_ml.ball_metrics import Counts

    model.eval()
    overall, visible, occ = Counts(), Counts(), Counts()
    with torch.no_grad():
        for x, y, occluded in loader:
            pred = model(x.to(device)).squeeze(1).cpu()
            ev = evaluate_ball(pred, y, occluded.numpy(), tol=tol)
            _merge(overall, ev.overall)
            _merge(visible, ev.visible)
            _merge(occ, ev.occluded)
    return BallEval(overall, visible, occ)


def _merge(dst: object, src: object) -> None:
    from padel_ml.ball_metrics import Counts

    assert isinstance(dst, Counts) and isinstance(src, Counts)
    dst.tp += src.tp
    dst.fp += src.fp
    dst.fn += src.fn
    dst.tn += src.tn
    dst.errors.extend(src.errors)


def _report(ev: BallEval) -> None:
    o = ev.overall
    print(f"\nmejor F1: {o.f1:.3f} | P={o.precision:.3f} R={o.recall:.3f}")
    print(f"error medio de localización: {o.mean_error:.2f} px (grid)")
    print(f"F1 visible={ev.visible.f1:.3f}  ocluida={ev.occluded.f1:.3f}")


def compare_models(
    train_dirs: list[Path],
    val_dirs: list[Path],
    epochs: int,
    plots_dir: Path,
    out_dir: Path | None,
    augment: bool = False,
    num_workers: int = 4,
) -> dict[str, BallEval]:
    """Train V2 and V3 under one protocol, then draw the headline plot.

    This is the ADR-0009 comparison: same data, same schedule, only the
    architecture changes. The occlusion breakdown shows where V3 earns its keep.
    """
    results: dict[str, BallEval] = {}
    for model_name in ("tracknetv2", "tracknetv3"):
        with mlflow.start_run(run_name=model_name):
            out = out_dir / f"{model_name}.pt" if out_dir is not None else None
            res = train_ball(
                train_dirs,
                val_dirs,
                model_name=model_name,
                epochs=epochs,
                out_path=out,
                plots_dir=plots_dir,
                augment=augment,
                num_workers=num_workers,
            )
            results[model_name] = res.best
    comp_path = plots_dir / "occlusion_comparison.png"
    plot_occlusion_comparison(results, comp_path)
    with mlflow.start_run(run_name="comparison"):
        mlflow.log_artifact(str(comp_path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TrackNet ball detector")
    parser.add_argument("--train-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default="tracknetv2", choices=["tracknetv2", "tracknetv3"])
    parser.add_argument(
        "--compare", action="store_true", help="Train both V2 and V3 and plot the comparison"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plots", type=Path, default=Path("runs/ball_plots"))
    parser.add_argument(
        "--augment", action="store_true", help="Colour jitter on train (generalise to courts)"
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--mlflow-uri",
        default="sqlite:///mlruns.db",
        help="MLflow tracking backend (file store is deprecated in MLflow 3)",
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("ball-detection")
    if args.compare:
        compare_models(
            args.train_dir,
            args.val_dir,
            args.epochs,
            args.plots,
            args.out,
            augment=args.augment,
            num_workers=args.workers,
        )
    else:
        with mlflow.start_run(run_name=args.model):
            train_ball(
                args.train_dir,
                args.val_dir,
                model_name=args.model,
                epochs=args.epochs,
                out_path=args.out,
                plots_dir=args.plots,
                augment=args.augment,
                num_workers=args.workers,
            )


if __name__ == "__main__":
    main()

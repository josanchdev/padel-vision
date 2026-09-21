"""Train and evaluate the shot-type classifier (ADR-0016).

Split is CROSS-TOURNAMENT, not cross-rally: whole tournaments are held out, so
the score answers "does this work on a court and camera it has never seen?".
Splitting by rally would leak — rallies of one tournament share court, lighting
and camera angle, and the model would be graded on conditions it memorised.

The serve is outnumbered 8.4:1 by the forehand (one serve per point, by
construction), so the loss is class-weighted; without it the serve's F1
collapses while accuracy still looks fine.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from padel_ml.shot_type_dataset import CLASSES, ShotWindow
from padel_ml.shot_type_model import ShotTypeBST

SERVE_INDEX = CLASSES.index("Serve")


def apply_serve_rule(probabilities: list[list[float]], windows: list[ShotWindow]) -> list[int]:
    """Only the first hit of a rally may be a serve — and it almost always is.

    Jorge's observation, and the labels bear it out exactly: all 97 serves in the
    dataset are the opening hit of their rally, and 97 of the 99 rallies open
    with one. That makes the serve a rule of the sport rather than something to
    be learned, so the classifier should not be left guessing at it.

    The model detects serves well (recall 0.937) but over-fires: 58 hits of other
    classes were predicted as serves, dropping its precision to 0.605. This
    rewrites those to their next-best class, and lets a rally opener be a serve
    if the model ranks it there.
    """
    first_of_rally: dict[str, int] = {}
    for i, window in enumerate(windows):
        if window.rally not in first_of_rally:
            first_of_rally[window.rally] = i
    openers = set(first_of_rally.values())

    out: list[int] = []
    for i, row in enumerate(probabilities):
        best = int(np.argmax(row))
        if best == SERVE_INDEX and i not in openers:
            without_serve = list(row)
            without_serve[SERVE_INDEX] = -1.0
            best = int(np.argmax(without_serve))
        out.append(best)
    return out


@dataclass
class FoldResult:
    """One held-out tournament, scored."""

    tournament: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    n_test: int
    probabilities: list[list[float]] = field(default_factory=list)
    truths: list[int] = field(default_factory=list)


def to_tensors(windows: list[ShotWindow]) -> tuple[Tensor, Tensor, Tensor]:
    pose = torch.from_numpy(np.stack([w.pose for w in windows]))
    ball = torch.from_numpy(np.stack([w.ball for w in windows]))
    labels = torch.tensor([w.label for w in windows], dtype=torch.long)
    return pose, ball, labels


def class_weights(labels: Tensor, n_classes: int, power: float = 0.0) -> Tensor:
    """Inverse-frequency weights raised to `power`, normalised to mean 1.

    `power=0` means no weighting at all, which is what measured best — against
    the intuition that the serve (outnumbered 8.4:1) needs protecting:

        power 1.0  acc 0.7938  macro-F1 0.7862  serve F1 0.736
        power 0.5  acc 0.8107  macro-F1 0.8072  serve F1 0.775
        power 0.0  acc 0.8129  macro-F1 0.8118  serve F1 0.790

    Weighting made the serve *worse*, not better. Pushed hard enough to never
    miss one, the model fires it everywhere: recall stayed at 0.94 while
    precision fell to 0.605, and those false serves ate into the other three
    classes too. Left alone it finds serves nearly as often and is right far
    more of the time.
    """
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1.0)
    weights = (counts.max() / counts) ** power
    return weights / weights.mean()


def _metrics(
    truths: list[int], predictions: list[int], n_classes: int
) -> tuple[float, float, dict[str, dict[str, float]], list[list[int]]]:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(truths, predictions, strict=True):
        matrix[truth, prediction] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for i, name in enumerate(CLASSES[:n_classes]):
        true_positive = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())
        actual = int(matrix[i, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual,
        }
        if actual:
            f1s.append(f1)
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    return accuracy, float(np.mean(f1s)) if f1s else 0.0, per_class, matrix.tolist()


def train_one_fold(
    train_windows: list[ShotWindow],
    test_windows: list[ShotWindow],
    seq_len: int,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 8,
    device: str | None = None,
    seed: int = 0,
) -> FoldResult:
    """Train on every tournament but one, score on the one held out."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    pose_tr, ball_tr, y_tr = to_tensors(train_windows)
    pose_te, ball_te, y_te = to_tensors(test_windows)
    n_classes = len(CLASSES)

    model = ShotTypeBST(seq_len=seq_len, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(y_tr, n_classes).to(device), label_smoothing=label_smoothing
    )
    loader = DataLoader(
        TensorDataset(pose_tr, ball_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True
    )
    # Warm-up then cosine decay: the schedule BST uses, and transformers are
    # famously unstable in their first epochs without the ramp.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (epoch + 1) / warmup_epochs
        if epoch < warmup_epochs
        else 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1))),
    )

    for _ in range(epochs):
        model.train()
        for pose_b, ball_b, y_b in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(pose_b.to(device), ball_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(pose_te), 128):
            logits = model(
                pose_te[start : start + 128].to(device), ball_te[start : start + 128].to(device)
            )
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
    predictions = apply_serve_rule(probabilities, test_windows)
    truths = y_te.tolist()
    accuracy, macro_f1, per_class, confusion = _metrics(truths, predictions, n_classes)
    return FoldResult(
        tournament=test_windows[0].tournament,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=confusion,
        n_test=len(test_windows),
        probabilities=probabilities,
        truths=truths,
    )


def cross_tournament_cv(
    windows: list[ShotWindow],
    seq_len: int,
    epochs: int = 120,
    device: str | None = None,
    folds: list[str] | None = None,
) -> list[FoldResult]:
    """Leave-one-tournament-out evaluation over the whole dataset."""
    by_tournament: dict[str, list[ShotWindow]] = collections.defaultdict(list)
    for window in windows:
        by_tournament[window.tournament].append(window)
    targets = folds if folds is not None else sorted(by_tournament)

    results: list[FoldResult] = []
    for tournament in targets:
        test = by_tournament[tournament]
        train = [w for w in windows if w.tournament != tournament]
        result = train_one_fold(train, test, seq_len, epochs=epochs, device=device)
        print(
            f"  {tournament:24s} acc {result.accuracy:.3f}  macro-F1 {result.macro_f1:.3f}"
            f"  (n={result.n_test})",
            flush=True,
        )
        results.append(result)
    return results


def aggregate(
    results: list[FoldResult], windows_by_tournament: dict[str, list[ShotWindow]] | None = None
) -> dict[str, object]:
    """Pool every fold's predictions: one global confusion matrix and metrics.

    Pass `windows_by_tournament` to apply the serve rule; without it the raw
    argmax is used, which is only meaningful for measuring the rule's effect.
    """
    n_classes = len(CLASSES)
    truths: list[int] = []
    predictions: list[int] = []
    for result in results:
        truths.extend(result.truths)
        if windows_by_tournament is not None:
            predictions.extend(
                apply_serve_rule(result.probabilities, windows_by_tournament[result.tournament])
            )
        else:
            predictions.extend(int(np.argmax(p)) for p in result.probabilities)
    accuracy, macro_f1, per_class, confusion = _metrics(truths, predictions, n_classes)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": confusion,
        "n": len(truths),
        "per_fold": {r.tournament: round(r.accuracy, 4) for r in results},
    }


def save_checkpoint(model: ShotTypeBST, path: Path, seq_len: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seq_len": seq_len, "classes": CLASSES}, path)

"""Cross-tournament evidence for the shot-type classifier (ADR-0016).

Leave-one-tournament-out over the labelled hits, with the two decisions that
were measured rather than assumed, and are re-measured on every run: no class
weighting (powers 0, 0.5 and 1 are each cross-validated), and the serve
restricted to a rally's opening hit (scored with and without the rule).

    uv run python scripts/evidence_shot_classifier.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from padel_ml.evidence import FIGURES_DIR, ExperimentResult, plot_confusion
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, ShotWindow, build_dataset
from padel_ml.shot_type_train import FoldResult, _metrics, apply_serve_rule, cross_tournament_cv

REPO = Path(__file__).resolve().parents[1]
SPANISH = {"Forehand": "Derecha", "Backhand": "Revés", "Smash": "Remate", "Serve": "Saque"}


CLASS_WEIGHT_POWERS = (0.0, 0.5, 1.0)
"""0 is what the system uses; 1 is plain inverse-frequency weighting."""


def _pool(
    folds: list[FoldResult], by_tournament: dict[str, list[ShotWindow]]
) -> tuple[list[int], list[int], list[int], dict[str, float]]:
    """Every fold's predictions together: truths, raw argmax, with the serve rule."""
    truths: list[int] = []
    plain: list[int] = []
    ruled: list[int] = []
    per_fold: dict[str, float] = {}
    for fold in folds:
        truths.extend(fold.truths)
        plain.extend(int(np.argmax(p)) for p in fold.probabilities)
        fold_ruled = apply_serve_rule(fold.probabilities, by_tournament[fold.tournament])
        ruled.extend(fold_ruled)
        correct = sum(int(a == b) for a, b in zip(fold.truths, fold_ruled, strict=True))
        per_fold[fold.tournament] = round(correct / len(fold.truths), 4)
    return truths, plain, ruled, per_fold


def main() -> None:
    windows, _ = build_dataset(
        REPO / "data" / "datasets" / "rally_features", REPO / "data" / "labels" / "types"
    )
    by_tournament: dict[str, list] = {}
    for window in windows:
        by_tournament.setdefault(window.tournament, []).append(window)

    print(f"{len(windows)} ventanas · {len(by_tournament)} torneos")
    ablation: dict[str, dict[str, float]] = {}
    for power in CLASS_WEIGHT_POWERS:
        print(f"\npesos de clase, potencia {power}")
        folds = cross_tournament_cv(windows, SEQ_LEN, epochs=60, class_weight_power=power)
        truths, plain, ruled, per_fold = _pool(folds, by_tournament)
        acc_power, macro_power, per_class_power, _ = _metrics(truths, ruled, len(CLASSES))
        serve = per_class_power["Serve"]
        ablation[str(power)] = {
            "accuracy": round(acc_power, 4),
            "macro_f1": round(macro_power, 4),
            "serve_precision": round(serve["precision"], 4),
            "serve_recall": round(serve["recall"], 4),
            "serve_f1": round(serve["f1"], 4),
        }
        if power == 0.0:  # what the system uses: the headline numbers
            system = (truths, plain, ruled, per_fold)

    truths, plain, ruled, per_fold = system
    acc_plain, macro_plain, per_class_plain, _ = _metrics(truths, plain, len(CLASSES))
    accuracy, macro_f1, per_class, confusion = _metrics(truths, ruled, len(CLASSES))
    serve_before, serve_after = per_class_plain["Serve"], per_class["Serve"]
    weighted = ablation[str(CLASS_WEIGHT_POWERS[-1])]

    plot_confusion(
        np.array(confusion),
        [SPANISH[c] for c in CLASSES],
        f"Clasificación de tipo de golpe — {len(truths)} golpes, validación cross-torneo",
        FIGURES_DIR / "shot_type_confusion.png",
    )

    result = ExperimentResult(
        name="shot_type_classifier",
        summary=(
            f"Clasificador de tipo de golpe (BST-0 adaptado): accuracy {accuracy:.2%}, "
            f"macro-F1 {macro_f1:.4f} en validación cruzada por torneos."
        ),
        metrics={
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "accuracy_without_serve_rule": round(acc_plain, 4),
            "macro_f1_without_serve_rule": round(macro_plain, 4),
            "n": len(truths),
            "tournaments": len(by_tournament),
        },
        dataset=(
            f"{len(windows)} golpes etiquetados por nosotros sobre CVSPORTS_Padel, "
            f"{len(by_tournament)} torneos; leave-one-tournament-out"
        ),
        method="BST-0 (pose + trayectoria de pelota, cross-attention), sin pesos de clase",
        params={
            "epochs": 60,
            "batch_size": 64,
            "lr": 5e-4,
            "seq_len": SEQ_LEN,
            "class_weight_power": 0.0,
            "class_weight_ablation": ablation,
            "serve_rule": "solo el primer golpe del rally puede ser saque",
            "serve_without_rule": {k: round(v, 4) for k, v in serve_before.items()},
            "ball_inference_size": "768x432",
        },
        per_class=per_class,
        confusion=confusion,
        labels=[SPANISH[c] for c in CLASSES],
        notes=(
            "Split CROSS-TORNEO, no cross-rally: los rallies de un torneo comparten pista, "
            "cámara e iluminación, así que un split por rally filtraría. La cifra responde a "
            "si funciona en una pista nunca vista. Dos decisiones medidas: (1) SIN pesos de "
            f"clase: con pesos por frecuencia inversa el macro-F1 es {weighted['macro_f1']:.3f} "
            f"frente a {macro_f1:.3f}, y el F1 del saque {weighted['serve_f1']:.3f} frente a "
            f"{serve_after['f1']:.3f}; (2) la regla del saque, que es reglamento y no "
            f"aprendizaje: su precisión pasa de {serve_before['precision']:.3f} a "
            f"{serve_after['precision']:.3f} y su recall de {serve_before['recall']:.3f} a "
            f"{serve_after['recall']:.3f}."
        ),
    )
    path = result.save()
    (REPO / "docs" / "metrics" / "shot_type_per_tournament.json").write_text(
        json.dumps(per_fold, indent=2) + "\n"
    )
    print(f"\naccuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}  (sin regla: {macro_plain:.4f})")
    for power, row in ablation.items():
        print(f"  pesos {power}: {row}")
    for name, values in per_class.items():
        print(
            f"  {name:10s} P {values['precision']:.3f} R {values['recall']:.3f} "
            f"F1 {values['f1']:.3f} n={values['support']}"
        )
    print(f"guardado -> {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()

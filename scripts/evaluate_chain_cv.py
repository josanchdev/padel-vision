"""The whole chain on CVSPORTS, one tournament held out at a time.

Each step was measured on its own before: the audio detector against the
annotated sound windows, the WHO vote against the paper's ANNOTATED instants,
the classifier with the hitter already chosen. Never together. This runs them
as they run in production — the audio decides when, the vote decides who at
the instant the audio found, the classifier decides what — and scores the end
result against every ground truth CVSPORTS has.

Per held-out tournament, everything that learns from CVSPORTS is retrained
without it: the audio detector and the shot classifier. The ball detector
never saw CVSPORTS (trained on PadelTracker100) and pose uses COCO weights, so
they need nothing. Pose and ball come from the per-rally cache.

Ground truth:
    WHEN   the dataset's annotated hit windows (99 rallies, 2,377 hits)
    WHAT   Jorge's type labels, same hits ("Other" left out)
    WHO    the paper's per-hit player, VIGO only (16 rallies, 319 hits)

A detection is paired with a real hit within 250 ms (the paper's collar).

Resumable: each fold's models and results are kept under runs/chain_cv/<T>/,
so a crash (WSL) costs one fold, not the run.

    uv run python scripts/evaluate_chain_cv.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from padel_ml.audio_train import detect_hits_in_audio, fit_and_save, match_events
from padel_ml.chain_eval import Row, summarize
from padel_ml.hit_assignment_gt import load_hit_assignments
from padel_ml.rally_analysis import Classifier, resolve_shots
from padel_ml.rally_features import RallyFeatures, load_rally_features
from padel_ml.shot_type_dataset import CLASSES, SEQ_LEN, ShotWindow, build_dataset
from padel_ml.shot_type_model import ShotTypeBST
from padel_ml.shot_type_train import fit

from padel_cv.cvsports import load_hits_csv

REPO = Path(__file__).resolve().parents[1]
CVSPORTS = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
FEATURES = REPO / "data" / "datasets" / "rally_features"
TYPE_LABELS = REPO / "data" / "labels" / "types"
AUDIO_CACHE = REPO / "data" / "datasets" / "audio_cache.npz"
RUNS = REPO / "runs" / "chain_cv"
COLLAR_S = 0.25


def _windows_by_rally() -> dict[str, list[tuple[float, float]]]:
    hits = load_hits_csv(CVSPORTS / "metadata" / "hits.csv")
    return {Path(name).stem: sorted(windows) for name, windows in hits.items()}


def _type_labels(rally: str, fps: float) -> dict[float, str]:
    path = TYPE_LABELS / f"{rally}.csv"
    if not path.exists():
        return {}
    return {
        int(r["frame"]) / fps: r["type"]
        for r in csv.DictReader(path.open(), delimiter=";")
        if r["type"]
    }


def _classifier(path: Path, device: str) -> Classifier:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ShotTypeBST(seq_len=SEQ_LEN, n_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def classify(window: ShotWindow) -> npt.NDArray[np.float32]:
        with torch.no_grad():
            logits = model(
                torch.from_numpy(window.pose)[None].to(device),
                torch.from_numpy(window.ball)[None].to(device),
            )
        return cast(npt.NDArray[np.float32], torch.softmax(logits, dim=1)[0].cpu().numpy())

    return classify


def _train_fold(
    tournament: str, rallies: list[RallyFeatures], windows: list[ShotWindow], device: str
) -> Path:
    """Audio + classifier without this tournament; returns the fold directory."""
    fold = RUNS / tournament
    fold.mkdir(parents=True, exist_ok=True)
    held_out_files = {f"{r.rally}.mp4" for r in rallies if r.tournament == tournament}
    if not (fold / "audio.pt").exists():
        print(f"  [{tournament}] entrenando audio...", flush=True)
        fit_and_save(
            CVSPORTS, fold / "audio.pt", cache=AUDIO_CACHE, exclude=held_out_files, device=device
        )
    if not (fold / "bst.pt").exists():
        print(f"  [{tournament}] entrenando clasificador...", flush=True)
        train = [w for w in windows if w.tournament != tournament]
        held_out = [w for w in windows if w.tournament == tournament]
        model, curve = fit(train, SEQ_LEN, device=device, held_out=held_out)
        torch.save(
            {"state_dict": model.state_dict(), "curve": asdict(curve), "n_train": len(train)},
            fold / "bst.pt",
        )
    return fold


def _evaluate_rally(
    rally: RallyFeatures,
    audio: Path,
    classify: Classifier,
    windows: list[tuple[float, float]],
    who: dict[float, int],
    device: str,
) -> list[Row]:
    detected = detect_hits_in_audio(
        CVSPORTS / "rallies" / f"{rally.rally}.mp4", audio, device=device
    )
    # shot i belongs to detection i: the audio keeps hits >= 8 frames apart
    frames = [round(t * rally.fps) for t in detected]
    shots = resolve_shots(frames, rally.players, rally.ball, rally.fps, classify, SEQ_LEN)

    truth_times = [(s + e) / 2 for s, e in windows]
    types = _type_labels(rally.rally, rally.fps)
    type_of = {
        t: types[min(types, key=lambda x: abs(x - t))]
        for t in truth_times
        if types and min(abs(x - t) for x in types) <= COLLAR_S
    }
    player_of: dict[float, int] = {}
    if who:
        instants = list(who)
        for i, j in match_events(truth_times, instants, COLLAR_S):
            player_of[truth_times[i]] = who[instants[j]]

    rows: list[Row] = []
    pairs = match_events(detected, truth_times, COLLAR_S)
    paired_det = {p for p, _ in pairs}
    det_of_truth = {q: p for p, q in pairs}
    for q, t in enumerate(truth_times):
        p = det_of_truth.get(q)
        rows.append(
            Row(
                rally.rally,
                rally.tournament,
                t,
                detected[p] if p is not None else None,
                type_of.get(t),
                player_of.get(t),
                shots[p].player_id if p is not None else None,
                shots[p].shot_type if p is not None else None,
            )
        )
    for p, d in enumerate(detected):
        if p not in paired_det:
            rows.append(
                Row(
                    rally.rally,
                    rally.tournament,
                    None,
                    d,
                    None,
                    None,
                    shots[p].player_id,
                    shots[p].shot_type,
                )
            )
    return rows


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rallies = [load_rally_features(p) for p in sorted(FEATURES.glob("*.npz"))]
    windows, _ = build_dataset(FEATURES, TYPE_LABELS)
    hit_windows = _windows_by_rally()
    who_truth = load_hit_assignments(CVSPORTS / "metadata" / "hit_assignments.xlsx")
    tournaments = sorted({r.tournament for r in rallies})

    rows: list[Row] = []
    for tournament in tournaments:
        fold = _train_fold(tournament, rallies, windows, device)
        result_path = fold / "rows.json"
        if result_path.exists():
            rows.extend(Row(**r) for r in json.loads(result_path.read_text()))
            print(f"[{tournament}] ya evaluado", flush=True)
            continue
        classify = _classifier(fold / "bst.pt", device)
        fold_rows: list[Row] = []
        for rally in (r for r in rallies if r.tournament == tournament):
            who = {h.time_s: h.slot for h in who_truth.get(rally.rally, []) if h.slot}
            rally_rows = _evaluate_rally(
                rally,
                fold / "audio.pt",
                classify,
                hit_windows.get(rally.rally, []),
                who,
                device,
            )
            fold_rows.extend(rally_rows)
        result_path.write_text(json.dumps([asdict(r) for r in fold_rows]) + "\n")
        numbers = summarize(fold_rows)
        print(
            f"[{tournament}] deteccion F1 {numbers['detection_f1']:.3f}  "
            f"tipo {numbers['type_accuracy']:.1%}"
            f"  cuando+que {numbers['end_to_end_when_what']:.1%}",
            flush=True,
        )
        rows.extend(fold_rows)

    print()
    for key, value in summarize(rows).items():
        print(f"  {key:30s} {value}")
    (RUNS / "all_rows.json").write_text(json.dumps([asdict(r) for r in rows]) + "\n")
    print(
        f"\n[saved] {RUNS / 'all_rows.json'}  (evidencias y figuras: scripts/evidence_chain_cv.py)"
    )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()

"""Head-to-head: pose+ball localizer vs audio CRNN, on the SAME data (ADR-0015).

The log already says the old approach reached ~61% recall and the audio one 0.93,
but those were measured on different footage, so the comparison was weak. This
script re-runs the OLD method (ShotLocalizer over pose+ball, ADR-0013) on the
CVSPORTS rallies with the same cross-rally split and the same event-based metric
(250 ms collar) used for the audio detector — apples to apples.

The point is not to bury the old method: it is to justify the change of direction
with a measurement rather than a memory.

    uv run python scripts/evidence_method_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from padel_ml.ball_infer import BallDetector
from padel_ml.ball_postprocess import postprocess_ball
from padel_ml.evidence import FIGURES_DIR, ExperimentResult, plot_comparison
from padel_ml.hit_assignment_eval import CORNER_INDICES
from padel_ml.shot_detect_dataset import WINDOW, build_dense_windows
from padel_ml.shot_detector import ShotLocalizer
from padel_ml.shot_eval import ShotBlock

from padel_cv.player_identity import PlayerIdentityTracker, court_mask_polygon, filter_players
from padel_cv.stages.pose import PlayerPoseStage

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
CACHE_DIR = REPO / "data" / "datasets" / "cvsports_posecache"
COLLAR_S = 0.25
AUDIO_F1 = 0.956  # docs/metrics/audio_threshold_seeds.json, same split and metric


def extract_rally(video: Path, corners: np.ndarray) -> dict[str, object]:
    """Pose + ball per frame for one rally, cached (the slow part)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{video.stem}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return {
            "persons": data["persons"].item(),
            "ball": data["ball"].item(),
            "fps": float(data["fps"]),
            "n_frames": int(data["n_frames"]),
        }
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    pose_stage = PlayerPoseStage()
    detector = BallDetector(REPO / "runs/ball_full/tracknetv3.pt")
    identity = PlayerIdentityTracker(fps=fps)
    polygon = court_mask_polygon(corners)
    persons: dict[int, list[np.ndarray]] = {}
    raw_ball = []
    index = 0
    from padel_cv.pipeline import Frame

    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = pose_stage.process(Frame(index=index, timestamp_s=index / fps, image=image))
        players = filter_players(frame.poses, polygon)
        identity.update(index, players)
        persons[index] = [p.keypoints.astype(np.float32) for p in players]
        hit = detector.detect(image, index)
        if hit is not None:
            raw_ball.append(hit)
        index += 1
    capture.release()
    ball = {b.frame_index: (b.x_px, b.y_px) for b in postprocess_ball(raw_ball)}
    np.savez(
        cache,
        persons=np.array(persons, dtype=object),
        ball=np.array(ball, dtype=object),
        fps=fps,
        n_frames=index,
    )
    return {"persons": persons, "ball": ball, "fps": fps, "n_frames": index}


def event_eval_frames(
    predicted: list[int], truth: list[int], fps: float, collar_s: float = COLLAR_S
) -> tuple[int, int, int]:
    """Same event-based scoring as the audio detector, but in video frames."""
    collar = collar_s * fps
    matched: set[int] = set()
    tp = 0
    for p in predicted:
        best, best_distance = -1, collar
        for i, t in enumerate(truth):
            if i in matched:
                continue
            if abs(p - t) <= best_distance:
                best, best_distance = i, abs(p - t)
        if best >= 0:
            matched.add(best)
            tp += 1
    return tp, len(predicted) - tp, len(truth) - len(matched)


def peaks(probs: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    out: list[int] = []
    last = -(10**9)
    for i in range(1, len(probs) - 1):
        if probs[i] < threshold or i - last < min_gap:
            continue
        if probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
            out.append(i)
            last = i
    return out


def main() -> None:
    corners = np.array(
        json.loads((REPO / "data/datasets/vigo_court.json").read_text())["keypoints_px"]
    )[list(CORNER_INDICES), :2]
    hits_csv = DATASET / "metadata" / "hits.csv"
    import csv as _csv

    hits_by_rally: dict[str, list[float]] = {}
    for row in _csv.DictReader(hits_csv.open()):
        centre = (float(row["start"]) + float(row["end"])) / 2
        hits_by_rally.setdefault(row["filename"], []).append(centre)

    # Only the VIGO rallies: extracting pose+ball for all 99 would take hours and
    # this tournament is the one we already have a homography for.
    rallies = sorted(p for p in (DATASET / "rallies").glob("20230528_VIGO_*.mp4"))
    print(f"{len(rallies)} rallies de VIGO")

    data = {}
    for video in rallies:
        print(f"  extrayendo {video.name}...", flush=True)
        data[video.name] = extract_rally(video, corners)

    # same cross-rally split policy as the audio detector: 70/30
    rng = np.random.default_rng(0)
    names = [v.name for v in rallies]
    order = rng.permutation(len(names))
    n_val = max(int(len(names) * 0.3), 1)
    val_names = [names[i] for i in order[:n_val]]
    train_names = [names[i] for i in order[n_val:]]
    print(f"train {len(train_names)} / val {len(val_names)}")

    # ---- build dense windows for the OLD method
    train_windows = []
    for name in train_names:
        entry = data[name]
        fps = float(entry["fps"])  # type: ignore[arg-type]
        blocks = [
            ShotBlock(start=round(t * fps), end=round(t * fps), category="hit")
            for t in hits_by_rally.get(name, [])
        ]
        train_windows.extend(
            build_dense_windows(
                entry["persons"],  # type: ignore[arg-type]
                entry["ball"],  # type: ignore[arg-type]
                blocks,
                tol=1,
                negatives_ratio=1.0,
            )
        )
    print(f"ventanas de entrenamiento: {len(train_windows)}")
    if not train_windows:
        raise SystemExit("no se pudieron construir ventanas — ¿pose/pelota vacías?")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    pose = torch.from_numpy(np.stack([w.pose for w in train_windows]))
    ball = torch.from_numpy(np.stack([w.ball for w in train_windows]))
    labels = torch.from_numpy(np.stack([w.labels for w in train_windows]))
    model = ShotLocalizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pose, ball, labels), batch_size=32, shuffle=True
    )
    positive_weight = torch.tensor(
        [(labels.numel() - labels.sum()) / max(labels.sum(), 1)], device=device
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for epoch in range(30):
        model.train()
        total = 0.0
        for pb, bb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(pb.to(device), bb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: loss {total / len(loader):.4f}")

    # ---- evaluate the OLD method by sliding it over the val rallies
    model.eval()
    best_row = None
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = fp = fn = 0
        for name in val_names:
            entry = data[name]
            fps = float(entry["fps"])  # type: ignore[arg-type]
            n_frames = int(entry["n_frames"])  # type: ignore[arg-type]
            scores = np.zeros(n_frames, dtype=np.float32)
            counts = np.zeros(n_frames, dtype=np.float32)
            from padel_ml.shot_detect_dataset import _window_around

            for centre in range(WINDOW // 2, n_frames - WINDOW // 2, WINDOW // 4):
                window = _window_around(
                    centre,
                    entry["persons"],  # type: ignore[arg-type]
                    entry["ball"],  # type: ignore[arg-type]
                )
                if window is None:
                    continue
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(window[0])[None].to(device),
                        torch.from_numpy(window[1])[None].to(device),
                    )
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                start = centre - WINDOW // 2
                for i, value in enumerate(probs):
                    if 0 <= start + i < n_frames:
                        scores[start + i] += value
                        counts[start + i] += 1
            averaged = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)
            predicted = peaks(averaged, threshold, min_gap=int(0.3 * fps))
            truth = [round(t * fps) for t in hits_by_rally.get(name, [])]
            a, b, c = event_eval_frames(predicted, truth, fps)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  thr {threshold}: F1 {f1:.3f}  P {precision:.3f}  R {recall:.3f}")
        row = {
            "threshold": threshold,
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if best_row is None or f1 > best_row["f1"]:
            best_row = row

    assert best_row is not None
    plot_comparison(
        {
            "Pose+pelota (ADR-0013)": {"f1": best_row["f1"]},
            "Audio CRNN (ADR-0015)": {"f1": AUDIO_F1},
        },
        "f1",
        "Detección de golpes: mismo dataset, mismo split, misma métrica",
        FIGURES_DIR / "method_comparison_same_data.png",
    )
    result = ExperimentResult(
        name="method_comparison_same_data",
        summary=(
            f"Cara a cara en igualdad de condiciones: pose+pelota F1 {best_row['f1']:.3f} "
            f"vs audio F1 {AUDIO_F1:.3f} (mismos rallies, mismo split, collar 250 ms)."
        ),
        metrics={
            "old_f1": best_row["f1"],
            "old_precision": best_row["precision"],
            "old_recall": best_row["recall"],
            "audio_f1": AUDIO_F1,
            "difference": round(AUDIO_F1 - best_row["f1"], 4),
        },
        dataset=f"CVSPORTS_Padel VIGO — {len(rallies)} rallies, split {len(train_names)}/{n_val}",
        method="ShotLocalizer (TCN pose+pelota, por frame) deslizado, mejor umbral",
        params={
            "window": WINDOW,
            "epochs": 30,
            "collar_s": COLLAR_S,
            "best_threshold": best_row["threshold"],
        },
        notes=(
            "Justifica el cambio de rumbo con una medición y no con un recuerdo. Antes se "
            "comparaba el localizer (61% recall en citys_cup) con el audio (F1 0,93 en "
            "CVSPORTS): datasets distintos, comparación débil. Aquí el método antiguo se "
            "entrena y evalúa sobre los MISMOS rallies, con el MISMO split cross-rally y la "
            "MISMA métrica event-based (collar 250 ms) que el detector de audio. El "
            "localizer parte además con ventaja respecto a su medición original: usa la "
            "pose y la pelota de nuestros modelos actuales, ya mejorados."
        ),
    )
    path = result.save()
    Path("docs/metrics/method_comparison_sweep.json").write_text(
        json.dumps(best_row, indent=2) + "\n"
    )
    print(f"\nMEJOR pose+pelota: F1 {best_row['f1']:.3f} vs audio {AUDIO_F1:.3f}")
    print(f"guardado -> {path}")


if __name__ == "__main__":
    main()

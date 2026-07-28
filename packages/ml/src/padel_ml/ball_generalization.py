"""Cross-court generalization report for the ball detector.

Runs the detector over several courts (WPT + new venues from YouTube) and
tabulates, per court, the detection rate (fraction of frames with a ball found)
and the mean confidence of those detections. Without ground truth on the new
courts we can't compute F1, but these two numbers quantify "how much it detects
and how sure it is" — the honest way to expose the generalization gap (ADR-0005).

Outputs a console table + JSON, a bar chart PNG for slides, and (optionally) a
verification mp4 per court for visual inspection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from padel_ml.ball_export import write_verification_video
from padel_ml.ball_infer import detect_ball_in_video


@dataclass
class CourtResult:
    court: str
    frames: int
    detections: int
    detection_rate: float  # detections / frames
    mean_confidence: float


def evaluate_court(
    name: str,
    video: Path,
    checkpoint: Path,
    min_confidence: float,
    max_frames: int | None,
    video_out: Path | None,
) -> CourtResult:
    if video_out is not None:
        hits = write_verification_video(video, checkpoint, video_out, min_confidence, max_frames)
    else:
        hits = detect_ball_in_video(video, checkpoint, min_confidence, max_frames)
    frames = max_frames if max_frames is not None else _count_frames(video)
    mean_conf = sum(h.confidence for h in hits) / len(hits) if hits else 0.0
    return CourtResult(
        court=name,
        frames=frames,
        detections=len(hits),
        detection_rate=len(hits) / frames if frames else 0.0,
        mean_confidence=mean_conf,
    )


def _count_frames(video: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def print_table(results: list[CourtResult]) -> None:
    print(f"\n{'court':<28} {'frames':>8} {'det.rate':>9} {'mean conf':>10}")
    print("-" * 58)
    for r in results:
        print(f"{r.court:<28} {r.frames:>8} {r.detection_rate:>8.1%} {r.mean_confidence:>10.3f}")


def plot_rates(results: list[CourtResult], out_path: Path) -> None:
    """Bar chart of detection rate per court — the generalization-gap figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r.court for r in results]
    rates = [r.detection_rate for r in results]
    bars = ax.bar(names, rates, color="#2563eb")
    for bar, r in zip(bars, results, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r.detection_rate + 0.01,
            f"{r.detection_rate:.0%}\nconf {r.mean_confidence:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("tasa de detección de pelota")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generalización del detector de pelota por pista")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_report(
    courts: dict[str, Path],
    checkpoint: Path,
    out_dir: Path,
    min_confidence: float = 0.5,
    max_frames: int | None = None,
    write_videos: bool = False,
) -> list[CourtResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CourtResult] = []
    for name, video in courts.items():
        video_out = out_dir / f"{name}_verif.mp4" if write_videos else None
        print(f"\n[{name}] running detector on {video.name}...")
        results.append(
            evaluate_court(name, video, checkpoint, min_confidence, max_frames, video_out)
        )
    print_table(results)
    (out_dir / "generalization.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    plot_rates(results, out_dir / "generalization.png")
    print(f"\nWrote report to {out_dir}/ (json + png" + (" + videos" if write_videos else "") + ")")
    return results

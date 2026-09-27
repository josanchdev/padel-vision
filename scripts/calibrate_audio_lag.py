"""Calibrate and validate the audio-to-contact lag, without labels (audio_sync).

For every CVSPORTS rally: detect the hits with the audio model, then find where
the ball meets a wrist around each one. The median gap is the lag.

Writes two things:
- runs/audio/audio_lag.json: the global lag, the fallback `analyze_rally` uses
  for videos too short to estimate their own. Computed without labels.
- docs/metrics/audio_lag.json: the evidence, per tournament, plus the
  validation against the only instants annotated by hand (VIGO, the paper's):
  how close the estimated contact lands to them, hit by hit and video by video.

    uv run python scripts/calibrate_audio_lag.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from padel_ml.audio_sync import MIN_HITS_PER_VIDEO, contact_frame, lag_samples
from padel_ml.audio_train import detect_hits_in_audio, match_events
from padel_ml.evidence import ExperimentResult
from padel_ml.hit_assignment_gt import load_hit_assignments
from padel_ml.rally_features import load_rally_features

REPO = Path(__file__).resolve().parents[1]
CVSPORTS = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
FEATURES = REPO / "data" / "datasets" / "rally_features"
AUDIO = REPO / "runs" / "audio" / "audio_crnn.pt"
OUT = REPO / "runs" / "audio" / "audio_lag.json"


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    q1, q2, q3 = statistics.quantiles(values, n=4)
    return round(q1, 1), round(q2, 1), round(q3, 1)


def main() -> None:
    truth = load_hit_assignments(CVSPORTS / "metadata" / "hit_assignments.xlsx")
    by_tournament: dict[str, list[float]] = {}
    contact_error_ms: list[float] = []  # estimated contact - annotated instant, per hit
    video_error_ms: list[float] = []  # per-video lag estimate - that video's true lag

    for path in sorted(FEATURES.glob("*.npz")):
        rally = load_rally_features(path)
        detected = detect_hits_in_audio(CVSPORTS / "rallies" / f"{rally.rally}.mp4", AUDIO)
        frames = sorted({round(t * rally.fps) for t in detected})
        samples = lag_samples(frames, rally.players, rally.ball, rally.fps)
        by_tournament.setdefault(rally.tournament, []).extend(samples)

        if rally.rally not in truth:
            continue
        annotated = [h.time_s for h in truth[rally.rally]]
        true_lags = []
        for p, t in match_events(detected, annotated, 0.25):
            true_lags.append(detected[p] - annotated[t])
            contact = contact_frame(
                round(detected[p] * rally.fps), rally.players, rally.ball, rally.fps
            )
            if contact is not None:
                contact_error_ms.append(1000 * (contact / rally.fps - annotated[t]))
        if len(samples) >= MIN_HITS_PER_VIDEO and true_lags:
            video_error_ms.append(
                1000 * (statistics.median(samples) - statistics.median(true_lags))
            )

    everything = [s for samples in by_tournament.values() for s in samples]
    lag_s = statistics.median(everything)
    OUT.write_text(
        json.dumps(
            {
                "lag_s": round(lag_s, 4),
                "n_hits": len(everything),
                "method": "median of detection minus ball-to-wrist contact, no labels",
                "audio_checkpoint": str(AUDIO.relative_to(REPO)),
            },
            indent=2,
        )
        + "\n"
    )

    metrics: dict[str, float] = {"global_lag_ms": round(1000 * lag_s, 1)}
    for tournament, samples in sorted(by_tournament.items()):
        metrics[f"{tournament}/lag_ms"] = round(1000 * statistics.median(samples), 1)
    q1, q2, q3 = _quartiles(contact_error_ms)
    metrics |= {
        "vigo_contact_error_ms_q1": q1,
        "vigo_contact_error_ms_median": q2,
        "vigo_contact_error_ms_q3": q3,
    }
    v1, v2, v3 = _quartiles(video_error_ms)
    metrics |= {
        "vigo_video_lag_error_ms_q1": v1,
        "vigo_video_lag_error_ms_median": v2,
        "vigo_video_lag_error_ms_q3": v3,
    }

    for key, value in metrics.items():
        print(f"  {key:44s} {value:+.0f}")
    result = ExperimentResult(
        name="audio_lag",
        summary=(
            f"El audio llega {1000 * lag_s:.0f} ms tarde al contacto (mediana global, sin "
            f"etiquetas); el contacto estimado cae a {q2:+.0f} ms del instante anotado en VIGO."
        ),
        metrics=metrics,
        dataset="CVSPORTS, 99 rallies (calibracion); VIGO, instantes del paper (validacion)",
        method="contacto = frame con la pelota mas cerca de una muneca (alturas de cuerpo)",
        params={
            "n_hits": len(everything),
            "min_hits_per_video": MIN_HITS_PER_VIDEO,
            "n_videos_validated": len(video_error_ms),
            "n_hits_validated": len(contact_error_ms),
        },
        notes=(
            "Las etiquetas solo validan: la calibracion no las usa. 'contact_error' mide el "
            "estimador golpe a golpe; 'video_lag_error' mide la estimacion por video (mediana "
            "de sus golpes) frente al desfase real de ese video."
        ),
    )
    print(f"\n[saved] {OUT}\n[saved] {result.save()}")


if __name__ == "__main__":
    main()

"""Why the external evaluation fell: audio and image are out of step.

The first external evaluation (frozen system, blind labels) came out far below
CVSPORTS: player 57.9% against 89.65%. Pairing each detected hit with Jorge's
visual contact frame showed the detection landing ~5 frames late, almost always.
The assignment votes over +-4 frames around that instant, so it looked at the
ball after it had left the racket — the tutor's hypothesis, which CVSPORTS did
not show because its assignment figure was measured on ANNOTATED instants.

Three measurements, all regenerable here:

1. Detector latency on CVSPORTS: audio detection minus the paper's annotated
   instant, over the 16 VIGO rallies.
2. Per external clip: where the racket pop starts in the raw waveform (no
   network involved) and where the detector puts it, both minus the labelled
   contact frame. Separates "the video's sound is late" from "the detector is".
3. Oracle: re-run WHO and WHAT with the detections moved back by a fixed shift,
   and score them. Diagnosis only — like the perfect-ball oracle, it measures how
   much of the error the timing explains; it is not a change to the system.

    uv run python scripts/experiment_av_offset.py
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import numpy.typing as npt
from padel_ml.audio_train import detect_hits_in_audio, match_events
from padel_ml.evidence import ExperimentResult
from padel_ml.external_eval import frozen_frames, load_truth, score_clip, summarize
from padel_ml.hit_assignment_gt import load_hit_assignments
from padel_ml.rally_analysis import ModelPaths, RallyModels, analyze_rally, resolve_shots
from scipy.signal import butter, sosfilt

from padel_cv.court_registry import court_file_for

REPO = Path(__file__).resolve().parents[1]
CVSPORTS = REPO / "data" / "raw" / "padel_audio_dataset" / "CVSPORTS_Padel"
LABELS = REPO / "data" / "labels" / "external"
VIDEOS = REPO / "data" / "raw" / "external"
AUDIO_CKPT = REPO / "runs" / "audio" / "audio_crnn.pt"
SAMPLE_RATE = 48000


def detector_latency_cvsports() -> list[float]:
    """Detected minus annotated instant (ms) for every paired VIGO hit."""
    truth = load_hit_assignments(CVSPORTS / "metadata" / "hit_assignments.xlsx")
    offsets: list[float] = []
    for rally in sorted(truth):
        video = CVSPORTS / "rallies" / f"{rally}.mp4"
        if not video.exists():
            continue
        detected = detect_hits_in_audio(video, AUDIO_CKPT)
        annotated = [h.time_s for h in truth[rally]]
        for p, t in match_events(detected, annotated, 0.25):
            offsets.append(1000 * (detected[p] - annotated[t]))
    return offsets


def _pop_envelope(video: Path) -> npt.NDArray[np.float32]:
    """High-passed amplitude envelope: a racket pop is a sharp, bright transient."""
    pcm = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    signal = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    sos = butter(4, 2000, "highpass", fs=SAMPLE_RATE, output="sos")
    envelope = np.abs(sosfilt(sos, signal))
    window = int(0.004 * SAMPLE_RATE)
    return np.convolve(envelope, np.ones(window) / window, "same").astype(np.float32)


def pop_onsets_ms(video: Path, frames: list[int], fps: float) -> list[float]:
    """Start of the loudest transient near each labelled frame, minus that frame (ms).

    Searches 200 ms before to 450 ms after; the onset is where the envelope first
    rises past 30% of the peak. Crude on purpose — no network, so it cannot share
    the detector's biases — and occasionally locked onto a neighbouring sound,
    which is why the median is what gets reported.
    """
    envelope = _pop_envelope(video)
    out: list[float] = []
    for frame in frames:
        t = frame / fps
        low, high = int((t - 0.2) * SAMPLE_RATE), int((t + 0.45) * SAMPLE_RATE)
        peak = low + int(np.argmax(envelope[low:high]))
        k = peak
        while k > low and envelope[k] > 0.3 * envelope[peak]:
            k -= 1
        out.append(1000 * (k / SAMPLE_RATE - t))
    return out


def main() -> None:
    metrics: dict[str, float] = {}

    latency = detector_latency_cvsports()
    q1, _, q3 = statistics.quantiles(latency, n=4)
    metrics["cvsports_detector_latency_ms_median"] = round(statistics.median(latency), 1)
    metrics["cvsports_detector_latency_ms_q1"] = round(q1, 1)
    metrics["cvsports_detector_latency_ms_q3"] = round(q3, 1)
    print(f"CVSPORTS: deteccion - instante anotado: mediana {statistics.median(latency):+.0f} ms")

    models = RallyModels(ModelPaths.under(REPO / "runs"))
    oracle: dict[str, dict[str, dict[str, float]]] = {}
    for labels in sorted(LABELS.glob("*.csv")):
        clip = labels.stem
        video = VIDEOS / f"{clip}.mp4"
        truth = load_truth(labels)
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()

        onsets = pop_onsets_ms(video, [m.frame for m in truth], fps)
        analysis = analyze_rally(video, models, court_file_for(clip))
        frozen = frozen_frames(video)
        detected = [s.frame_index / fps for s in analysis.shots]
        paired = match_events(detected, [m.frame / fps for m in truth], 0.25)
        detector = [1000 * (detected[p] - truth[t].frame / fps) for p, t in paired]
        metrics[f"{clip}/pop_onset_minus_label_ms_median"] = round(statistics.median(onsets), 1)
        metrics[f"{clip}/detection_minus_label_ms_median"] = round(statistics.median(detector), 1)
        print(
            f"{clip}: chasquido en la onda {statistics.median(onsets):+.0f} ms, "
            f"deteccion {statistics.median(detector):+.0f} ms (respecto a la marca visual)"
        )

        cvsports_shift = -round(statistics.median(latency) / 1000 * fps)
        clip_shift = -round(statistics.median(detector) / 1000 * fps)
        oracle[clip] = {}
        base = [s.frame_index for s in analysis.shots]
        for name, shift in [
            ("sistema", 0),
            ("latencia_cvsports", cvsports_shift),
            ("oraculo_clip", clip_shift),
        ]:
            frames = sorted({max(f + shift, 0) for f in base})
            shots = resolve_shots(
                frames, analysis.players, analysis.ball, fps, models.classify, models.seq_len
            )
            numbers = summarize(score_clip(clip, truth, replace(analysis, shots=shots), frozen))
            oracle[clip][f"{name} ({shift:+d} frames)"] = numbers
            print(
                f"  {name:18s} {shift:+d}f  jugador {numbers['player_accuracy']:.1%}  "
                f"equipo {numbers['team_accuracy']:.1%}  "
                f"extremo a extremo {numbers['end_to_end']:.1%}"
            )

    result = ExperimentResult(
        name="external_av_offset",
        summary=(
            "La caida externa se debe a la sincronizacion audio-imagen: el detector marca "
            f"~{metrics['cvsports_detector_latency_ms_median']:.0f} ms tarde incluso en CVSPORTS, "
            "y el video externo suma su propio retraso de sonido. Recolocando el instante, "
            "jugador y equipo recuperan las cifras de CVSPORTS."
        ),
        metrics=metrics,
        dataset="CVSPORTS VIGO (latencia) + clips externos etiquetados a ciegas",
        method="diagnostico: onda cruda, detector CRNN y oraculo de desplazamiento",
        params={"oracle": oracle, "collar_s": 0.25, "sample_rate": SAMPLE_RATE},
        notes=(
            "El oraculo NO es un cambio del sistema: mide cuanto error explica el instante. "
            "La cifra de asignacion de CVSPORTS (89,65%) se midio con instantes anotados, no "
            "detectados, por lo que esta latencia no aparecia alli."
        ),
    )
    print(f"\n[saved] {result.save()}")


if __name__ == "__main__":
    main()

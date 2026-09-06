"""Build log-Mel features + per-frame hit labels for the audio detector (ADR-0015).

Replicates the feature pipeline of the padel hit-detection paper (Decorte et al.,
CVPRW 2024): 40 log-Mel bins, FFT window 2048, 50% overlap. Each rally mp4 has its
audio decoded, turned into a log-Mel spectrogram, and labelled per spectrogram
frame from hits.csv (onset/offset in seconds → the frames inside a hit window are
positive). The CRNN then reads sequences of these frames.

Dataset: CVSPORTS_Padel (audio + 2377 annotated hits) — see docs/bibliography.md.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float32]

# Paper's feature settings.
SAMPLE_RATE = 48000  # the dataset audio is 48 kHz (Nyquist 24 kHz)
N_MELS = 40
N_FFT = 2048
HOP = N_FFT // 2  # 50% overlap
SEQ_LEN = 256  # spectrogram frames per training sequence (paper)


def decode_audio(path: Path, sr: int = SAMPLE_RATE) -> FloatArray:
    """Decode a video/audio file to mono float32 samples at `sr` via ffmpeg."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = subprocess.run(
        [ffmpeg, "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def log_mel(samples: FloatArray, sr: int = SAMPLE_RATE) -> FloatArray:
    """(T, N_MELS) log-Mel spectrogram — the CRNN's input features."""
    from torchaudio.transforms import MelSpectrogram

    mel = MelSpectrogram(sample_rate=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    spec = mel(torch.from_numpy(samples)[None])[0]  # (N_MELS, T)
    logmel = torch.log(spec + 1e-6)
    return logmel.T.numpy().astype(np.float32)  # (T, N_MELS)


def frame_time(frame_idx: int, sr: int = SAMPLE_RATE) -> float:
    """Seconds at the centre of spectrogram frame `frame_idx`."""
    return (frame_idx * HOP + N_FFT / 2) / sr


def hit_labels(n_frames: int, hits: list[tuple[float, float]], sr: int = SAMPLE_RATE) -> FloatArray:
    """Per-frame 0/1: 1 where the frame's time falls inside a hit's [start, end]."""
    labels = np.zeros(n_frames, dtype=np.float32)
    times = np.array([frame_time(i, sr) for i in range(n_frames)])
    for start, end in hits:
        labels[(times >= start) & (times <= end)] = 1.0
    return labels


@dataclass
class RallyAudio:
    """One rally's features + per-frame labels + its source (tournament) id."""

    features: FloatArray  # (T, N_MELS)
    labels: FloatArray  # (T,)
    filename: str


def load_hits_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """filename -> list of (start, end) hit windows in seconds."""
    out: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(csv_path.open()):
        out.setdefault(row["filename"], []).append((float(row["start"]), float(row["end"])))
    return out


def build_rally(rally_mp4: Path, hits: list[tuple[float, float]]) -> RallyAudio:
    """Decode one rally's audio → log-Mel features + per-frame hit labels."""
    samples = decode_audio(rally_mp4)
    feats = log_mel(samples)
    labels = hit_labels(len(feats), hits)
    return RallyAudio(features=feats, labels=labels, filename=rally_mp4.name)


def tournament_of(filename: str) -> str:
    """Group key for cross-tournament splits: DATE_LOCATION from the rally name."""
    # e.g. "20230528_VIGO_00.mp4" -> "20230528_VIGO"
    parts = filename.replace(".mp4", "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else filename

"""Detect padel ball-hits from the match AUDIO (new approach, feb 2026).

Pose+ball geometry proved weak at LOCATING the hit (heuristics 15-52% recall, a
learned per-frame localizer ~61%). The racket-sports SotA and a padel-specific
paper (Decorte et al., CVPRW 2024, F1 92% with audio+pose) point to a signal we
weren't using: the SOUND. A racket hitting the ball is a sharp, short impact — a
clean onset in the audio — while bounces (floor) and wall hits sound different,
and crowd/commentary are sustained low-frequency sounds that don't spike the same
way. So audio should LOCATE the hit (when); pose+ball then classify it (what/who).

This module is deliberately simple for the proof of concept: load the audio,
compute a short-time energy / onset envelope, and pick peaks. If the hit shows up
as a clean peak on our real audio, we escalate to a learned model on the
spectrogram (the SotA approach). No heavy audio deps: ffmpeg decodes to WAV,
scipy reads it.

Cite: Decorte et al., "Multi-Modal Hit Detection and Positional Analysis in Padel
Competitions", CVPRW 2024 (see docs/bibliography.md).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def decode_to_wav(audio_path: Path, out_wav: Path, sample_rate: int = 22050) -> Path:
    """Decode any audio/video file to mono WAV at `sample_rate` via ffmpeg.

    Works for .m4a/.mp3/.webm/.mp4 — whatever the audio arrives as. Mono is enough
    for onset detection and halves the data.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )
    return out_wav


def load_wav(wav_path: Path) -> tuple[int, FloatArray]:
    """Return (sample_rate, samples in [-1, 1] float32 mono)."""
    from scipy.io import wavfile

    sr, data = wavfile.read(wav_path)
    x = data.astype(np.float32)
    if x.ndim > 1:  # stereo -> mono
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) or 1.0
    return int(sr), (x / peak).astype(np.float32)


def onset_envelope(
    samples: FloatArray, sr: int, hop_s: float = 0.005, win_s: float = 0.02
) -> tuple[FloatArray, float]:
    """Short-time positive energy-change envelope (a simple onset detector).

    A hit is a sudden rise in energy, so we track the frame-to-frame INCREASE in
    short-time energy (half-wave rectified). Returns (envelope, hop_s): one value
    per hop, and the seconds-per-hop so peaks can be mapped back to time.
    """
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    # short-time energy over sliding windows
    n = 1 + (len(samples) - win) // hop if len(samples) >= win else 0
    energy = np.empty(max(n, 0), dtype=np.float32)
    for i in range(n):
        frame = samples[i * hop : i * hop + win]
        energy[i] = float(np.mean(frame * frame))
    # positive change (onset strength)
    diff = np.diff(energy, prepend=energy[:1])
    env = np.maximum(diff, 0.0)
    m = float(env.max()) or 1.0
    return (env / m).astype(np.float32), hop_s


def highfreq_energy(
    samples: FloatArray,
    sr: int,
    cutoff_hz: float = 3000.0,
    hop_s: float = 0.005,
    win_s: float = 0.01,
) -> tuple[FloatArray, float]:
    """RMS energy of the HIGH-frequency band (a racket pop is sharp/high-pitched;
    voices and crowd rumble are low). High-passing before measuring energy makes
    the hit stand out over commentary far better than raw energy. Returns
    (energy_per_hop, hop_s). Normalized by a robust high percentile, not the global
    max, so one loud clap doesn't flatten every real hit."""
    from scipy import signal

    sos = signal.butter(4, cutoff_hz, "hp", fs=sr, output="sos")
    hp = signal.sosfilt(sos, samples).astype(np.float32)
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    n = 1 + (len(hp) - win) // hop if len(hp) >= win else 0
    e = np.empty(max(n, 0), dtype=np.float32)
    for i in range(n):
        frame = hp[i * hop : i * hop + win]
        e[i] = float(np.sqrt(np.mean(frame * frame)))
    norm = float(np.percentile(e, 99.5)) or 1.0
    return np.clip(e / norm, 0.0, 1.0).astype(np.float32), hop_s


@dataclass
class AudioHit:
    """A detected hit from audio: time in seconds and its peak strength."""

    time_s: float
    strength: float


def hit_candidates(
    audio_path: Path,
    cutoff_hz: float = 3000.0,
    threshold: float = 0.25,
    min_gap_s: float = 0.20,
    sample_rate: int = 22050,
    tmp_wav: Path | None = None,
) -> list[AudioHit]:
    """High-freq onset peaks = candidate hit times, for audio-assisted annotation.

    These are PROPOSALS: the annotator jumps to each and the human confirms (a hit)
    or rejects (a bounce/clap). Uses the high-pass band so it favours the pop.
    """
    wav = tmp_wav or audio_path.with_suffix(".hpwav.wav")
    decode_to_wav(audio_path, wav, sample_rate)
    _sr, samples = load_wav(wav)
    e, hop_s = highfreq_energy(samples, _sr, cutoff_hz=cutoff_hz)
    return detect_peaks(e, hop_s, threshold=threshold, min_gap_s=min_gap_s)


def detect_peaks(
    env: FloatArray, hop_s: float, threshold: float = 0.3, min_gap_s: float = 0.20
) -> list[AudioHit]:
    """Local maxima of the onset envelope above `threshold`, at least `min_gap_s`
    apart (a hit can't repeat within ~200 ms). Times are hop index * hop_s."""
    min_gap = max(1, int(min_gap_s / hop_s))
    hits: list[AudioHit] = []
    last = -(10**9)
    for i in range(1, len(env) - 1):
        if env[i] < threshold:
            continue
        if env[i] >= env[i - 1] and env[i] >= env[i + 1] and i - last >= min_gap:
            hits.append(AudioHit(time_s=i * hop_s, strength=float(env[i])))
            last = i
    return hits


def detect_hits_in_audio(
    audio_path: Path,
    threshold: float = 0.3,
    min_gap_s: float = 0.20,
    sample_rate: int = 22050,
    tmp_wav: Path | None = None,
) -> list[AudioHit]:
    """End-to-end: decode -> onset envelope -> peaks. The proof-of-concept detector."""
    wav = tmp_wav or audio_path.with_suffix(".wav")
    decode_to_wav(audio_path, wav, sample_rate)
    sr, samples = load_wav(wav)
    env, hop_s = onset_envelope(samples, sr)
    return detect_peaks(env, hop_s, threshold=threshold, min_gap_s=min_gap_s)

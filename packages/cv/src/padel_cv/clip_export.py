"""Cut a short clip around a frame from a processed video (on-demand).

The web's shot table opens a modal with the clip of each shot (3s before / 3s
after) from the PROCESSED video — skeleton and label baked in. Rather than
pre-generate a file per shot (wasteful), we cut the window on demand with the
ffmpeg binary bundled by imageio-ffmpeg.

Seeking by time (fps * frame_index) keeps this independent of the source codec.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def cut_clip(
    video_path: Path,
    output_path: Path,
    center_frame: int,
    fps: float,
    margin_s: float = 3.0,
) -> Path:
    """Write a clip spanning [center_frame - margin, center_frame + margin].

    Times are derived from fps so the cut matches the frame index regardless of
    the container. Re-encodes to H.264 + faststart so the browser plays it inline.
    """
    center_s = center_frame / fps
    start_s = max(center_s - margin_s, 0.0)
    duration_s = margin_s * 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",  # no audio track needed for analysis clips
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path

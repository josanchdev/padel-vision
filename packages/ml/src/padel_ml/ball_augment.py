"""Colour/photometric augmentation for ball-detector training.

The ball model overfits to the courts it trained on (blue WPT). To generalise to
other courts and lighting, we jitter brightness, contrast and hue at train time,
simulating different venues without labelling new footage (ADR-0009/ADR-0005).

Two rules specific to this task:
- Photometric only, no geometry: the ball's position must stay exact (the
  heatmap target is not transformed), so we never shift/scale/flip the image.
- The SAME jitter is applied to all INPUT_FRAMES of a window, so the movement
  cue between frames (which is how TrackNet finds the ball) stays coherent.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def augment_window(window: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Apply one random photometric jitter to a (T, 3, H, W) float[0,1] window.

    The same parameters hit every frame. Returns a new array in [0, 1].
    """
    brightness = 1.0 + rng.uniform(-0.25, 0.25)
    contrast = 1.0 + rng.uniform(-0.25, 0.25)
    hue_shift = rng.uniform(-0.05, 0.05)  # fraction of the colour wheel

    out = window * brightness
    mean = out.mean(axis=(0, 2, 3), keepdims=True)  # per-channel mean over T,H,W
    out = (out - mean) * contrast + mean
    out = _shift_hue(out, hue_shift)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _shift_hue(window: FloatArray, shift: float) -> FloatArray:
    """Rotate hue by `shift` (in [0,1]) on a (T, 3, H, W) BGR window.

    Cheap approximation via the YIQ chroma plane rotation — no per-pixel HSV
    round-trip, so it stays fast enough for on-the-fly augmentation.
    """
    if abs(shift) < 1e-3:
        return window
    b, g, r = window[:, 0], window[:, 1], window[:, 2]
    angle = shift * 2.0 * np.pi
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Luma stays; rotate the two chroma components (I, Q in YIQ).
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    i2 = i * cos_a - q * sin_a
    q2 = i * sin_a + q * cos_a
    r2 = y + 0.956 * i2 + 0.621 * q2
    g2 = y - 0.272 * i2 - 0.647 * q2
    b2 = y - 1.106 * i2 + 1.703 * q2
    return np.stack([b2, g2, r2], axis=1)

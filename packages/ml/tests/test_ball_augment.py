import numpy as np
from padel_ml.ball_augment import augment_window


def test_output_stays_in_range() -> None:
    rng = np.random.default_rng(0)
    window = rng.random((3, 3, 8, 8)).astype(np.float32)
    out = augment_window(window, rng)
    assert out.shape == window.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.dtype == np.float32


def test_augmentation_changes_colours() -> None:
    rng = np.random.default_rng(1)
    window = np.full((3, 3, 8, 8), 0.5, dtype=np.float32)
    out = augment_window(window, rng)
    # A uniform grey should be shifted (brightness/contrast/hue), not identical.
    assert not np.allclose(out, window)


def test_same_jitter_across_frames() -> None:
    # Three identical frames must stay identical after augmentation (the same
    # parameters hit each frame, preserving the motion cue between them).
    rng = np.random.default_rng(2)
    frame = rng.random((3, 8, 8)).astype(np.float32)
    window = np.stack([frame, frame, frame])
    out = augment_window(window, rng)
    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[1], out[2])

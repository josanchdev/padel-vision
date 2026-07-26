import numpy as np
from padel_ml.augment import augment_clip, flip_horizontal, rotate_scale


def clip_with_marked_joints() -> np.ndarray:
    """(3, T, 17): put left shoulder (5) at x=+1, right shoulder (6) at x=-1."""
    clip = np.zeros((3, 4, 17), dtype=np.float32)
    clip[0, :, 5] = 1.0  # left shoulder x
    clip[0, :, 6] = -1.0  # right shoulder x
    clip[2] = 0.9  # confidence
    return clip


def test_flip_swaps_left_right_and_negates_x() -> None:
    flipped = flip_horizontal(clip_with_marked_joints())
    # After flip, what was left shoulder (x=+1) becomes the right-shoulder slot
    # with negated x, so slot 6 now holds x=-(+1)=-1 -> stays a valid skeleton.
    assert flipped[0, 0, 6] == -1.0  # left(+1) moved to slot 6, negated
    assert flipped[0, 0, 5] == 1.0  # right(-1) moved to slot 5, negated


def test_flip_is_involutive() -> None:
    clip = clip_with_marked_joints()
    np.testing.assert_allclose(flip_horizontal(flip_horizontal(clip)), clip)


def test_flip_preserves_confidence() -> None:
    flipped = flip_horizontal(clip_with_marked_joints())
    np.testing.assert_allclose(flipped[2], 0.9)


def test_rotate_scale_leaves_confidence_untouched() -> None:
    clip = clip_with_marked_joints()
    out = rotate_scale(clip, max_deg=20, max_scale=0.2, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[2], clip[2])  # confidence row unchanged


def test_rotate_zero_is_identity_on_xy() -> None:
    clip = np.random.rand(3, 4, 17).astype(np.float32)
    # max_deg=0 -> theta=0, max_scale=0 -> scale=1, so (x, y) are unchanged.
    out = rotate_scale(clip, max_deg=0, max_scale=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:2], clip[:2], atol=1e-6)


def test_augment_preserves_shape() -> None:
    clip = clip_with_marked_joints()
    out = augment_clip(clip, np.random.default_rng(1))
    assert out.shape == clip.shape

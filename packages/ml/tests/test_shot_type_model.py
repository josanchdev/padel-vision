import torch
from padel_ml.shot_type_model import TCN, ShotTypeBST, sinusoidal_encoding


def test_output_shape_is_one_logit_per_class() -> None:
    model = ShotTypeBST(seq_len=100, n_classes=4)
    out = model(torch.randn(3, 100, 17, 3), torch.randn(3, 100, 3))
    assert out.shape == (3, 4)


def test_gradients_reach_every_parameter() -> None:
    """A disconnected branch would train silently as dead weight."""
    model = ShotTypeBST(seq_len=40, n_classes=4)
    out = model(torch.randn(2, 40, 17, 3), torch.randn(2, 40, 3))
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 3])).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_tcn_keeps_the_sequence_length() -> None:
    """Padding must compensate the dilation, or frames silently disappear."""
    tcn = TCN(9, [16, 16], kernel_size=5)
    assert tcn(torch.randn(2, 9, 57)).shape == (2, 16, 57)


def test_positional_encoding_is_distinct_per_step() -> None:
    encoding = sinusoidal_encoding(10, 16)[0]
    assert encoding.shape == (10, 16)
    assert not torch.allclose(encoding[0], encoding[1])


def test_the_ball_changes_the_prediction() -> None:
    """The ball must reach the output: it is the input BST shows matters most."""
    model = ShotTypeBST(seq_len=30, n_classes=4).eval()
    pose = torch.randn(1, 30, 17, 3)
    with torch.no_grad():
        a = model(pose, torch.zeros(1, 30, 3))
        b = model(pose, torch.randn(1, 30, 3))
    assert not torch.allclose(a, b, atol=1e-5)

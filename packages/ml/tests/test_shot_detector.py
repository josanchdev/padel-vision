import torch
from padel_ml.shot_detector import ShotDetector


def test_forward_shapes() -> None:
    model = ShotDetector()
    pose = torch.randn(4, 32, 17, 3)
    ball = torch.randn(4, 32, 3)
    out = model(pose, ball)
    assert out.shape == (4,)  # one logit per window


def test_variable_window_length() -> None:
    # TCN keeps sequence length via padding, so a different T still works.
    model = ShotDetector()
    out = model(torch.randn(2, 20, 17, 3), torch.randn(2, 20, 3))
    assert out.shape == (2,)


def test_can_learn_a_ball_cued_signal() -> None:
    """The model must have the capacity to learn 'ball near the origin at the
    centre frame = shot'. Overfit a tiny synthetic set; loss should drop clearly."""
    torch.manual_seed(0)
    n, t = 32, 32
    pose = torch.randn(n, t, 17, 3) * 0.1  # pose carries no signal here
    ball = torch.randn(n, t, 3) * 0.1
    y = torch.zeros(n)
    y[: n // 2] = 1.0
    # Positives: ball sits at the origin (wrist) at the centre frame, present-flag on.
    ball[: n // 2, t // 2, :2] = 0.0
    ball[: n // 2, t // 2, 2] = 1.0
    # Negatives: ball far away at the centre.
    ball[n // 2 :, t // 2, :2] = 5.0
    ball[n // 2 :, t // 2, 2] = 1.0

    model = ShotDetector(d_model=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    first = loss_fn(model(pose, ball), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(pose, ball), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5, "the detector should learn the ball-cued signal"

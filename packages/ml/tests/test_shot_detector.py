import torch
from padel_ml.shot_detector import ShotDetector, ShotTypeClassifier


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


def test_type_classifier_shapes_with_and_without_ball() -> None:
    pose = torch.randn(4, 32, 17, 3)
    ball = torch.randn(4, 32, 3)
    with_ball = ShotTypeClassifier(n_classes=6, use_ball=True)
    without = ShotTypeClassifier(n_classes=6, use_ball=False)
    assert with_ball(pose, ball).shape == (4, 6)
    assert without(pose, ball).shape == (4, 6)  # ball ignored, still valid


def test_pose_only_has_no_ball_stream() -> None:
    m = ShotTypeClassifier(n_classes=6, use_ball=False)
    assert m.tcn_ball is None
    # And it must run even if handed a ball tensor (it just ignores it).
    out = m(torch.randn(2, 32, 17, 3), torch.randn(2, 32, 3))
    assert out.shape == (2, 6)


def test_ball_changes_the_prediction() -> None:
    # With use_ball on, the ball input must influence the output (else the ablation
    # is meaningless). Same pose, two different balls -> different logits.
    torch.manual_seed(0)
    m = ShotTypeClassifier(n_classes=6, use_ball=True)
    m.eval()
    pose = torch.randn(1, 32, 17, 3)
    a = m(pose, torch.zeros(1, 32, 3))
    b = m(pose, torch.ones(1, 32, 3))
    assert not torch.allclose(a, b)

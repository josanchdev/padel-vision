import torch
from padel_ml.audio_detector import AudioHitCRNN, focal_bce_loss


def test_crnn_per_frame_output() -> None:
    m = AudioHitCRNN(n_mels=40)
    out = m(torch.randn(2, 256, 40))
    assert out.shape == (2, 256)  # one logit per input frame


def test_crnn_variable_length() -> None:
    m = AudioHitCRNN(n_mels=40)
    assert m(torch.randn(1, 100, 40)).shape == (1, 100)


def test_focal_loss_positive_and_lower_when_right() -> None:
    logits_wrong = torch.tensor([[-3.0, -3.0]])  # predicts no-hit
    logits_right = torch.tensor([[3.0, 3.0]])  # predicts hit
    targets = torch.tensor([[1.0, 1.0]])
    lw = focal_bce_loss(logits_wrong, targets)
    lr = focal_bce_loss(logits_right, targets)
    assert lw > lr > 0


def test_crnn_learns_synthetic_hit() -> None:
    # a "hit" = high energy in one mel band at specific frames; model must learn it
    torch.manual_seed(0)
    n, t = 16, 64
    x = torch.randn(n, t, 40) * 0.1
    y = torch.zeros(n, t)
    x[:, 30, 10] += 5.0  # spike at frame 30 for all samples
    y[:, 30] = 1.0
    m = AudioHitCRNN(n_mels=40)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    first = focal_bce_loss(m(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = focal_bce_loss(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5

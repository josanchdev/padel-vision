"""CRNN for audio hit detection — SED-net replica (ADR-0015, Decorte et al. 2024).

Per-frame binary hit detector over log-Mel features. Architecture replicated from
the paper's Table 2 (adapted SED-net, ~109k params):

  conv2D 1->64 (3x3) ReLU, maxpool (1x5)     # pool FREQUENCY only, keep time
  conv2D 64->64 (3x3) ReLU, maxpool (1x2)
  conv2D 64->64 (3x3) ReLU, maxpool (1x2)
  reshape -> features per time step
  bidirectional GRU -> 32
  bidirectional GRU -> 16
  time-distributed dense 16 -> 16
  time-distributed dense 16 -> 1 (sigmoid, per-frame hit prob)

Pooling is over the mel axis only (1xk), never over time, so the output stays one
prediction per input frame — that's what makes it a precise per-frame localizer.
Trained with binary focal cross-entropy (the paper's loss) for the class imbalance.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class AudioHitCRNN(nn.Module):
    """log-Mel (B, T, n_mels) -> per-frame hit logits (B, T)."""

    def __init__(self, n_mels: int = 40, cnn_ch: int = 64, gru1: int = 32, gru2: int = 16) -> None:
        super().__init__()
        # CNN over the (mel x time) "image"; pool only the mel axis (dim=freq).
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((5, 1)),  # (freq, time) -> shrink freq by 5, keep time
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        # after pooling freq: n_mels/5/2/2. features per time step = cnn_ch * that.
        freq_out = n_mels // 5 // 2 // 2
        feat = cnn_ch * max(freq_out, 1)
        self.gru1 = nn.GRU(feat, gru1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(gru1 * 2, gru2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru2 * 2, gru2 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(gru2 * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_mels) -> (B, 1, mel, T) for conv (freq as height, time as width)
        b, t, _m = x.shape
        img = x.transpose(1, 2).unsqueeze(1)  # (B, 1, mel, T)
        c = self.cnn(img)  # (B, ch, freq', T)
        c = c.permute(0, 3, 1, 2).reshape(b, t, -1)  # (B, T, ch*freq')
        c, _ = self.gru1(c)
        c, _ = self.gru2(c)
        return cast(torch.Tensor, self.head(c).squeeze(-1))  # (B, T) logits


def focal_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Binary focal cross-entropy (the paper's loss). Down-weights easy negatives
    so the rare hit frames aren't drowned by the background."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.mean()

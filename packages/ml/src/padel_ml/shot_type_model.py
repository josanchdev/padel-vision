"""Shot-type classifier: BST-0 adapted to padel (ADR-0016 A).

Reimplemented from Chang's BST (CVPRW 2026, arXiv:2502.21085) and the TemPose
blocks it builds on, following the published architecture and the author's
reference implementation. Hyper-parameters are theirs: d_model 100, d_head 128,
6 heads, 2 temporal layers + 1 interaction layer, dropout 0.3, TCN kernel 5.

Two deliberate departures from the original, both because padel is not
badminton singles:

- **One player, not two.** BST feeds both players because in singles the
  opponent's position is informative about the rally. Here the hit has already
  been attributed to a specific player (86.8% team accuracy, ADR-0015 D) and
  what we classify is *that player's gesture*, so the model sees the hitter and
  the ball. The cross-attention between pose and ball — the part BST shows
  matters most — is kept intact.
- **The ball carries a presence flag.** TrackNet finds the ball in ~89% of
  frames, so a third channel marks which positions are real; without it a
  missing ball reads as a ball at the origin.

The flow: pose and ball each go through their own TCN, a shared temporal
transformer encodes each stream, cross-attention lets the pose query the ball
trajectory, and a final encoder summarises the pair before the MLP head.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

POSE_JOINTS = 17
POSE_CHANNELS = 3
"""x, y and keypoint confidence."""
BALL_CHANNELS = 3
"""x, y and a present/missing flag."""


def sinusoidal_encoding(length: int, dim: int) -> Tensor:
    """Standard 1-D positional encoding (BST uses the same, fixed not learned)."""
    position = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return encoding.unsqueeze(0)


class MLP(nn.Module):
    """Linear-GELU-dropout-linear, the block TemPose uses everywhere."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, drop_p: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class TCN(nn.Module):
    """Dilated temporal convolutions that keep the sequence length.

    Each layer widens its dilation, so a few layers already see a good chunk of
    the window — which is what lets the transformer above work on strokes rather
    than on raw frames.
    """

    def __init__(
        self, in_channels: int, channels: list[int], kernel_size: int = 5, drop_p: float = 0.3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, out_channels in enumerate(channels):
            previous = in_channels if i == 0 else channels[i - 1]
            dilation = i * 2 + 1
            padding = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(previous, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(drop_p),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention, depending on what is passed as key/value."""

    def __init__(self, d_model: int, d_head: int, n_head: int, drop_p: float) -> None:
        super().__init__()
        inner = d_head * n_head
        self.n_head = n_head
        self.scale = d_head**-0.5
        self.to_q = nn.Linear(d_model, inner, bias=False)
        self.to_kv = nn.Linear(d_model, inner * 2, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
        self.tail = nn.Sequential(nn.Linear(inner, d_model), nn.Dropout(drop_p))

    def forward(self, query_in: Tensor, context: Tensor | None = None) -> Tensor:
        context = query_in if context is None else context
        batch, length, _ = query_in.shape
        query = self.to_q(query_in).view(batch, length, self.n_head, -1).transpose(1, 2)
        key, value = (
            t.transpose(1, 2)
            for t in self.to_kv(context)
            .view(batch, -1, self.n_head, 2 * query.shape[-1])
            .chunk(2, dim=-1)
        )
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attended = self.attend(dots) @ value
        return cast(Tensor, self.tail(attended.transpose(1, 2).reshape(batch, length, -1)))


class TransformerLayer(nn.Module):
    """Pre-norm self-attention + feed-forward, both residual."""

    def __init__(self, d_model: int, d_head: int, n_head: int, hidden: int, drop_p: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, d_model, hidden, drop_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(self.norm1(x)) + x
        return cast(Tensor, self.feed_forward(self.norm2(x)) + x)


class CrossTransformerLayer(nn.Module):
    """Pose asks the ball trajectory what it was doing (BST's key idea)."""

    def __init__(self, d_model: int, d_head: int, n_head: int, hidden: int, drop_p: float) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.norm_out = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, d_model, hidden, drop_p)

    def forward(self, query_in: Tensor, context: Tensor) -> Tensor:
        attended = self.attention(self.norm_query(query_in), self.norm_context(context))
        return cast(Tensor, self.feed_forward(self.norm_out(attended)) + attended)


class ShotTypeBST(nn.Module):
    """Pose + ball -> shot type, for a single hitter.

    Input shapes: pose (B, T, 17, 3), ball (B, T, 3). Output: (B, n_classes).
    """

    def __init__(
        self,
        seq_len: int,
        n_classes: int = 4,
        d_model: int = 100,
        d_head: int = 128,
        n_head: int = 6,
        depth_temporal: int = 2,
        depth_interaction: int = 1,
        drop_p: float = 0.3,
        mlp_scale: int = 4,
        tcn_kernel: int = 5,
    ) -> None:
        super().__init__()
        hidden = d_model * mlp_scale
        self.tcn_pose = TCN(POSE_JOINTS * POSE_CHANNELS, [d_model, d_model], tcn_kernel, drop_p)
        self.tcn_ball = TCN(BALL_CHANNELS, [d_model // 2, d_model], tcn_kernel, drop_p)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.register_buffer("pos_temporal", sinusoidal_encoding(1 + seq_len, d_model))
        self.register_buffer("pos_cross", sinusoidal_encoding(seq_len, d_model))
        self.dropout = nn.Dropout(drop_p)
        self.encoder_temporal = nn.ModuleList(
            TransformerLayer(d_model, d_head, n_head, hidden, drop_p) for _ in range(depth_temporal)
        )
        self.cross = CrossTransformerLayer(d_model, d_head, n_head, hidden, drop_p)
        self.token_interaction = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.register_buffer("pos_interaction", sinusoidal_encoding(1 + seq_len, d_model))
        self.encoder_interaction = nn.ModuleList(
            TransformerLayer(d_model, d_head, n_head, hidden, drop_p)
            for _ in range(depth_interaction)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3), MLP(d_model * 3, n_classes, hidden, drop_p)
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight)

    def _encode(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Prepend the class token, encode, and split summary from the sequence."""
        batch = sequence.shape[0]
        token = self.class_token.expand(batch, -1, -1)
        x = torch.cat([token, sequence], dim=1) + cast(Tensor, self.pos_temporal)
        x = self.dropout(x)
        for layer in self.encoder_temporal:
            x = layer(x)
        return x[:, 0], x[:, 1:]

    def forward(self, pose: Tensor, ball: Tensor) -> Tensor:
        batch, length = pose.shape[0], pose.shape[1]
        pose_seq = self.tcn_pose(pose.reshape(batch, length, -1).transpose(1, 2)).transpose(1, 2)
        ball_seq = self.tcn_ball(ball.transpose(1, 2)).transpose(1, 2)

        pose_summary, pose_tokens = self._encode(pose_seq)
        ball_summary, ball_tokens = self._encode(ball_seq)

        # The gesture queries the trajectory: a smash and a defensive lift look
        # alike in the skeleton alone but send the ball in opposite directions.
        pos_cross = cast(Tensor, self.pos_cross)
        fused = self.cross(pose_tokens + pos_cross, ball_tokens + pos_cross)
        token = self.token_interaction.expand(batch, -1, -1)
        fused = torch.cat([token, fused], dim=1) + cast(Tensor, self.pos_interaction)
        for layer in self.encoder_interaction:
            fused = layer(fused)

        combined = torch.cat([pose_summary, ball_summary, fused[:, 0]], dim=1)
        return cast(Tensor, self.head(combined))

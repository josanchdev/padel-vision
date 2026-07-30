"""ShotDetector: a learned pose+ball temporal model for shot LOCALIZATION.

Modelo 1 of ADR-0013. Replaces the heuristic (ADR-0012, 15-52% recall) with a
model that learns "this pose-and-ball motion over time is a strike" — the way a
human recognises a hit without a fixed rule.

Design follows the pose+ball line of racket-sports SotA (TemPose CVPR 2023, BST
CVPR 2026), scaled down for our modest dataset (~1800 windows): a temporal
convolution (TCN) over each stream separately, then fusion, then a binary head.
A TCN (not a transformer) keeps the parameter count low; dilated 1D convs give a
wide temporal receptive field so the model sees the whole swing, not one frame.

Input per window (T frames):
- pose (N, T, 17, 3): normalized skeleton (x, y, confidence).
- ball (N, T, 3):     ball (x, y) in the skeleton's frame + present-flag.
Output: one logit (shot vs no-shot).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class TemporalConvBlock(nn.Module):
    """Dilated 1D conv over time, keeping the sequence length (odd kernel).

    Dilation widens the receptive field without extra layers, so a stack of a few
    blocks already spans the whole ~32-frame window.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, drop_p: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(drop_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class TCN(nn.Module):
    """A small stack of dilated temporal conv blocks (channels-first: N, C, T)."""

    def __init__(self, in_ch: int, channels: list[int], kernel_size: int = 5, drop_p: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 * i + 1
            layers.append(TemporalConvBlock(prev, ch, kernel_size, dilation, drop_p))
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class ShotDetector(nn.Module):
    """Pose+ball → shot/no-shot logit."""

    def __init__(
        self,
        n_joints: int = 17,
        joint_dim: int = 3,
        ball_dim: int = 3,
        d_model: int = 64,
        drop_p: float = 0.3,
    ) -> None:
        super().__init__()
        pose_in = n_joints * joint_dim
        # Each stream gets its own TCN, then we fuse the pooled summaries. Keeping
        # the streams separate (as BST does) lets each learn its own dynamics
        # before they meet.
        self.tcn_pose = TCN(pose_in, [d_model, d_model], drop_p=drop_p)
        self.tcn_ball = TCN(ball_dim, [d_model // 2, d_model], drop_p=drop_p)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
        )

    def forward(self, pose: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        # pose: (N, T, 17, 3) -> (N, 51, T); ball: (N, T, 3) -> (N, 3, T)
        n, t = pose.shape[0], pose.shape[1]
        pose_seq = pose.reshape(n, t, -1).transpose(1, 2)
        ball_seq = ball.transpose(1, 2)

        pose_feat = self.tcn_pose(pose_seq).mean(dim=-1)  # (N, d_model), temporal avg pool
        ball_feat = self.tcn_ball(ball_seq).mean(dim=-1)  # (N, d_model)
        fused = torch.cat([pose_feat, ball_feat], dim=1)
        return cast(torch.Tensor, self.head(fused).squeeze(-1))  # (N,)


class ShotTypeClassifier(nn.Module):
    """Pose(+ball) → stroke-type logits (Modelo 2, ADR-0013).

    Shares the detector's architecture (TCN per stream + fusion) but outputs one
    logit per stroke class. The `use_ball` flag is the whole point: with it off,
    the model is pose-only; with it on, it fuses the ball trajectory. Training the
    same network both ways ISOLATES what the ball contributes — the like-for-like
    comparison that answers "does the ball disambiguate the shot type?" (the
    scientific core, replacing the pose-only baseline's ~0.60 macro-F1).
    """

    def __init__(
        self,
        n_classes: int,
        *,
        use_ball: bool = True,
        n_joints: int = 17,
        joint_dim: int = 3,
        ball_dim: int = 3,
        d_model: int = 64,
        drop_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_ball = use_ball
        pose_in = n_joints * joint_dim
        self.tcn_pose = TCN(pose_in, [d_model, d_model], drop_p=drop_p)
        self.tcn_ball = TCN(ball_dim, [d_model // 2, d_model], drop_p=drop_p) if use_ball else None
        head_in = 2 * d_model if use_ball else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, pose: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        n, t = pose.shape[0], pose.shape[1]
        pose_feat = self.tcn_pose(pose.reshape(n, t, -1).transpose(1, 2)).mean(dim=-1)
        if self.tcn_ball is not None:
            ball_feat = self.tcn_ball(ball.transpose(1, 2)).mean(dim=-1)
            feat = torch.cat([pose_feat, ball_feat], dim=1)
        else:
            feat = pose_feat
        return cast(torch.Tensor, self.head(feat))  # (N, n_classes)

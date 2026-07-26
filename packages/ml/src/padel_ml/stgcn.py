"""A compact ST-GCN for padel shot classification.

Input: (batch, C=3, T=32, V=17) — normalized skeleton clips. Each block does a
spatial graph convolution (mix connected joints via the fixed adjacency) then a
temporal convolution (mix each joint across time), with residual connections.
Kept small on purpose: the dataset is modest (~1.3k clips), so a large model
would overfit; ST-GCN's built-in body structure lets it learn from little data.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from padel_ml.graph import NUM_JOINTS, normalized_adjacency


class SpatialGraphConv(nn.Module):
    """Graph convolution over the fixed skeleton adjacency."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        adjacency = torch.from_numpy(normalized_adjacency())
        self.register_buffer("adjacency", adjacency)
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V). Mix joints: sum_v' A[v,v'] x[...,v'].
        x = self.linear(x)
        return torch.einsum("nctv,vw->nctw", x, self.adjacency)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels and stride == 1:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.spatial_bn(self.spatial(x)))
        x = self.temporal(x)
        return cast(torch.Tensor, self.relu(x + res))


class STGCN(nn.Module):
    """Small ST-GCN: input (N, 3, T, 17) -> logits (N, num_classes)."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, 32),
                STGCNBlock(32, 32),
                STGCNBlock(32, 64, stride=2),
                STGCNBlock(64, 64),
                STGCNBlock(64, 128, stride=2),
            ]
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 1, 3, 2).reshape(n, c * v, t)
        x = self.input_bn(x).reshape(n, c, v, t).permute(0, 1, 3, 2)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3))  # global average over time and joints
        return cast(torch.Tensor, self.head(x))

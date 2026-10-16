"""Write the ball network table of section 4.2.3: memoria/figuras/tabla-red-pelota.tex.

The network the system uses (padel_ml.tracknet_v3: a reduced U-Net plus the
corrector) against the U-Net in its original configuration (Ronneberger et al.,
2015, §2 and fig. 1), on the three things the text says were changed and the
parameters they save.

Both sides come from code. Ours is measured with one forward pass at the size
the system feeds it. The original paper gives its layers (23 convolutions; 64
channels doubled at each of four poolings up to 1024, two 3x3 convolutions per
level, a 2x2 up-convolution per step back and a final 1x1) but not a parameter
count, so it is built here, with our input (three stacked RGB frames) and
output (one heatmap), and counted.

    uv run python scripts/build_ball_network_table.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from padel_ml.thesis.latex import count, tabular
from padel_ml.tracknet_v3 import TrackNetV3
from torch import nn

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "tabla-red-pelota.tex"

INPUT_W, INPUT_H = 768, 432
"""The frame size `BallDetector` feeds the network."""
IN_CHANNELS = 9
"""Three RGB frames, as the system stacks them."""
OUT_CHANNELS = 1
"""One heatmap."""
UNET_CHANNELS = (64, 128, 256, 512, 1024)
"""Channels per level in the original U-Net."""


def unet_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with ReLU, the original's unit (no batch norm)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(),
    )


def original_unet() -> nn.ModuleList:
    """The layers of the original U-Net (their weights; the wiring adds none)."""
    levels = UNET_CHANNELS
    layers: list[nn.Module] = [unet_block(IN_CHANNELS, levels[0])]
    layers += [unet_block(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]
    for i in range(len(levels) - 1, 0, -1):
        layers.append(nn.ConvTranspose2d(levels[i], levels[i - 1], 2, stride=2))
        layers.append(unet_block(levels[i], levels[i - 1]))  # after the skip concatenation
    layers.append(nn.Conv2d(levels[0], OUT_CHANNELS, 1))
    return nn.ModuleList(layers)


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def measure_ours(model: TrackNetV3) -> dict[str, int]:
    """Halvings down, doublings up and channels at both ends, from a forward pass."""
    unet = model.backbone
    widths: dict[str, int] = {}
    channels: dict[str, int] = {}

    def keep(name: str) -> object:
        def hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
            channels[name], widths[name] = int(output.shape[1]), int(output.shape[3])

        return hook

    for name in ("enc1", "bottleneck", "head"):
        getattr(unet, name).register_forward_hook(keep(name))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, INPUT_H, INPUT_W))
    return {
        "down": round(math.log2(INPUT_W / widths["bottleneck"])),
        "up": round(math.log2(widths["head"] / widths["bottleneck"])),
        "first": channels["enc1"],
        "deepest": channels["bottleneck"],
    }


def millions(value: int, places: int) -> str:
    """31035201 -> 31\\,M; 1761410 -> 1,76\\,M (the text says «1,76 millones»)."""
    return f"{value / 1e6:.{places}f}".replace(".", ",") + r"\,M"


def build_table() -> str:
    model = TrackNetV3()
    ours = measure_ours(model)
    steps = len(UNET_CHANNELS) - 1
    rows: list[list[str] | str] = [
        ["Número de veces que se reduce la imagen", str(steps), str(ours["down"])],
        ["Número de veces que se amplía la imagen", str(steps), str(ours["up"])],
        [
            "Rango de canales",
            rf"${UNET_CHANNELS[0]} \to {count(UNET_CHANNELS[-1])}$",
            rf"${ours['first']} \to {ours['deepest']}$",
        ],
        [
            "Parámetros",
            millions(parameters(original_unet()), 0),
            millions(parameters(model), 2),
        ],
    ]
    return tabular("lcc", ["", "U-Net", "Red propia"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

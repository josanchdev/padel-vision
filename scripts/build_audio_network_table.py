"""Write the layer table of the audio hit detector: memoria/figuras/tabla-red-audio.tex.

Built from the network itself (padel_ml.audio_detector.AudioHitCRNN), in the usual
layout for an architecture: layer type, hyperparameters and activation. Shapes are
measured with a forward pass over one sequence of 256 spectrogram frames.
Convolutions and pools are written channels x mel bands x frames; the rest, frames
x features. The network returns logits, and the sigmoid of the last layer is
applied when the probability is read (a sigmoid inside the loss, and in
`hit_probabilities`).

    uv run python scripts/build_audio_network_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from padel_ml.audio_dataset import N_MELS, SEQ_LEN
from padel_ml.audio_detector import AudioHitCRNN
from padel_ml.thesis.latex import tabular
from torch import nn

from padel_cv.paths import REPO

OUT = REPO / "memoria" / "figuras" / "tabla-red-audio.tex"


def build_table() -> str:
    model = AudioHitCRNN()
    shapes: dict[int, tuple[int, ...]] = {}

    def remember(module: nn.Module, _inputs: object, output: object) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        shapes[id(module)] = tuple(tensor.shape)  # type: ignore[union-attr]

    modules = [*model.cnn, model.gru1, model.gru2, *model.head]
    for module in modules:
        module.register_forward_hook(remember)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, SEQ_LEN, N_MELS))

    def image(module: nn.Module) -> str:  # channels x bands x frames
        _, channels, bands, frames = shapes[id(module)]
        return rf"{channels} \times {bands} \times {frames}"

    def sequence(shape: tuple[int, ...]) -> str:  # frames x features
        return rf"{shape[1]} \times {shape[2]}"

    rows: list[list[str] | str] = []
    previous: tuple[int, ...] = (1, 1, N_MELS, SEQ_LEN)
    for module in model.cnn:
        if isinstance(module, nn.Conv2d):
            kernel, stride = module.kernel_size, module.stride
            setting = (
                rf"${module.in_channels} \to {module.out_channels}$, "
                rf"${kernel[0]} \times {kernel[1]}$, ${stride[0]} \times {stride[1]}$"
            )
            rows.append(["conv2D", setting, "ReLU"])
        elif isinstance(module, nn.MaxPool2d):
            size = module.kernel_size
            pool = size if isinstance(size, tuple) else (size, size)
            rows.append(["maxpool", rf"${pool[0]} \times {pool[1]}$", "-"])
    # After the three blocks the bands are folded into the features of each frame
    _, channels, bands, frames = shapes[id(model.cnn[8])]
    folded = (1, frames, channels * bands)
    rows.append(
        [
            "reorganización",
            rf"${channels} \times {bands} \times {frames} \to {sequence(folded)}$",
            "-",
        ]
    )
    previous = folded
    for gru in (model.gru1, model.gru2):
        out = shapes[id(gru)]
        rows.append(
            [
                "GRU bidireccional",
                rf"${sequence(previous)} \to {sequence(out)}$",
                "tanh",
            ]
        )
        previous = out
    for index, layer in enumerate(model.head):
        if isinstance(layer, nn.Linear):
            out = shapes[id(layer)]
            activation = "ReLU" if index == 0 else "sigmoide"
            rows.append(
                [
                    "densa en el tiempo",
                    rf"${sequence(previous)} \to {sequence(out)}$",
                    activation,
                ]
            )
            previous = out
    return tabular("lcc", ["Tipo de capa", "Hiperparámetros", "Activación"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

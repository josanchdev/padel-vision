"""Write the layer table of the audio hit detector: memoria/figuras/tabla-red-audio.tex.

Built from the network itself (padel_ml.audio_detector.AudioHitCRNN): each layer's
configuration, its trainable parameters and its output shape for one sequence of
256 spectrogram frames, as measured with a forward pass. Convolutions and pools
show channels x mel bands x frames; the recurrent and dense layers, frames x
features.

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
    shapes: dict[str, tuple[int, ...]] = {}

    def remember(name: str):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            shapes[name] = tuple(tensor.shape)  # type: ignore[union-attr]

        return hook

    layers: list[tuple[str, nn.Module, str]] = [
        ("conv1", model.cnn[0], r"Conv2D $1 \to 64$, $3 \times 3$, ReLU"),
        ("pool1", model.cnn[2], r"MaxPool $5 \times 1$ (frecuencia)"),
        ("conv2", model.cnn[3], r"Conv2D $64 \to 64$, $3 \times 3$, ReLU"),
        ("pool2", model.cnn[5], r"MaxPool $2 \times 1$"),
        ("conv3", model.cnn[6], r"Conv2D $64 \to 64$, $3 \times 3$, ReLU"),
        ("pool3", model.cnn[8], r"MaxPool $2 \times 1$"),
        ("gru1", model.gru1, r"GRU bidireccional, 32 unidades"),
        ("gru2", model.gru2, r"GRU bidireccional, 16 unidades"),
        ("dense1", model.head[0], r"Densa $32 \to 32$, ReLU"),
        ("dense2", model.head[2], r"Densa $32 \to 1$ (sigmoide)"),
    ]
    for name, module, _ in layers:
        module.register_forward_hook(remember(name))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, SEQ_LEN, N_MELS))

    rows: list[list[str] | str] = []
    for name, module, text in layers:
        shape = shapes[name]
        if name.startswith(("conv", "pool")):  # (batch, channels, mel, time)
            out = rf"${shape[1]} \times {shape[2]} \times {shape[3]}$"
        else:  # (batch, time, features)
            out = rf"${shape[1]} \times {shape[2]}$"
        parameters = sum(p.numel() for p in module.parameters())
        rows.append([text, out, f"{parameters:,}".replace(",", r"\,") if parameters else "0"])
    total = sum(p.numel() for p in model.parameters())
    rows.append([r"\textbf{Total}", "", rf"\textbf{{{total:,}}}".replace(",", r"\,")])
    return tabular(
        "lrr",
        ["Capa", "Salida", "Parámetros"],
        rows,
        rules=[len(rows) - 1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

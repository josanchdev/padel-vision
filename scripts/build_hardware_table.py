"""Write the equipment table of section 5.3: memoria/figuras/tabla-equipo.tex.

The machine the processing cost was measured on, read from the machine itself
(padel_ml.thesis.hardware). It refuses to write the table on another machine:
the GPU has to be the one processing_time.json recorded
(scripts/evidence_processing_time.py).

    uv run python scripts/build_hardware_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from padel_ml.thesis.hardware import query_machine
from padel_ml.thesis.latex import tabular

from padel_cv.paths import METRICS, REPO

OUT = REPO / "memoria" / "figuras" / "tabla-equipo.tex"


def build_table() -> str:
    measured_on = json.loads((METRICS / "processing_time.json").read_text())["params"]["gpu"]
    machine = query_machine()
    if machine.gpu != measured_on:
        raise SystemExit(f"this GPU is {machine.gpu}; the cost was measured on {measured_on}")
    rows: list[list[str] | str] = [
        ["Tarjeta gráfica", f"{machine.gpu}, {machine.gpu_memory_gb}~GB"],
        ["Procesador", f"{machine.cpu}, {machine.cores} núcleos y {machine.threads} hilos"],
        ["Memoria RAM", f"{machine.memory_gb}~GB"],
        ["Sistema operativo", f"{machine.windows} con WSL2 ({machine.linux})"],
        [
            "Software",
            f"Python {machine.python}, PyTorch {machine.torch} (CUDA {machine.cuda})",
        ],
    ]
    return tabular("ll", ["Componente", "Especificación"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build_table() + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

"""The tables of the dissertation, as booktabs tabulars (memoria/GUIA.md, section 7).

Each function returns only the `tabular`: the chapter writes the `table`
environment, the caption and the label around `\\input{...}`, so the prose stays
in the chapter and the numbers stay here.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from padel_ml.thesis.latex import (
    count,
    percent,
    percent_range,
    proportion,
    proportion_range,
    quantity,
    tabular,
)
from padel_ml.thesis.sources import Evidence

CI = "IC 95\\,\\%"
SECONDS = r"\second"
PAPER = "\\cite{decorte2024padel}"


def steps_table(evidence: Evidence) -> str:
    """Table 1: each step measured on its own, next to Decorte et al."""
    audio = evidence("audio_threshold_seeds")["metrics"]
    assignment = evidence("hit_assignment_replica")["metrics"]

    def with_interval(key: str) -> str:
        value, low, high = evidence.interval(key)
        return f"{percent(value)} ({percent_range(low, high)})"

    return tabular(
        "lcc",
        ["Paso", "Este trabajo", f"Decorte et al.~{PAPER}"],
        [
            ["Cuándo: detección (F1)",
             f"{proportion(audio['f1'])} $\\pm$ {proportion(audio['f1_std'])}",
             proportion(audio["paper_f1"], 2)],
            ["Quién: jugador correcto", with_interval("tabla1_asignacion.player"),
             percent(assignment["paper_player_accuracy"])],
            ["Quién: pareja correcta", with_interval("tabla1_asignacion.team"),
             percent(assignment["paper_team_accuracy"])],
            ["Qué tipo: \\emph{accuracy}", with_interval("tabla1_tipo.accuracy"),
             "\\emph{no lo hace}"],
        ],
    )  # fmt: skip


def system_table(evidence: Evidence) -> str:
    """Table 2: the whole chain, each tournament judged by models that never saw it.

    Who hit is only annotated in VIGO, so those rows are measured there alone; the
    rows say over which hits each percentage is taken (padel_ml.chain_eval)."""

    def row(label: str, key: str) -> list[str]:
        value, low, high = evidence.interval(f"tabla2_cadena.{key}")
        if key == "detection_f1":
            return [f"\\quad {label}", f"{proportion(value)} ({proportion_range(low, high)})"]
        return [f"\\quad {label}", f"{percent(value)} ({percent_range(low, high)})"]

    return tabular(
        "lc",
        ["Medida", f"Valor ({CI})"],
        [
            "Los 11 torneos",
            row("Detección (F1)", "detection_f1"),
            row("Tipo correcto, sobre los detectados", "type_accuracy"),
            row("Detectado y con el tipo correcto", "end_to_end_when_what"),
            "VIGO, el único torneo con el jugador anotado",
            row("Jugador correcto, sobre los detectados", "player_accuracy"),
            row("Pareja correcta, sobre los detectados", "team_accuracy"),
            row("Todo correcto: detectado, jugador y tipo", "end_to_end_vigo"),
        ],
        rules=[4],
    )


def cost_table(evidence: Evidence) -> str:
    """Time per stage of `analyze_rally` on one rally, slowest first."""
    metrics = evidence("processing_time")["metrics"]
    total = metrics["total_s"]
    stages = [
        ("Pose de los jugadores", metrics["pose_s"]),
        ("Pelota", metrics["pelota_s"]),
        ("Resto: vídeo, identidad, voto y tipo", metrics["resto_s"]),
        ("Audio: detector de golpes", metrics["audio_s"]),
    ]
    rows: list[list[str] | str] = [
        [label, quantity(seconds, 1, SECONDS), percent(seconds / total)]
        for label, seconds in sorted(stages, key=lambda stage: -stage[1])
    ]
    rows.append(["Total", quantity(total, 1, SECONDS), ""])
    return tabular("lcc", ["Etapa", "Tiempo", "Parte del total"], rows, rules=[len(stages)])


def datasets_table(evidence: Evidence) -> str:
    """The three sources of data and what each one is used for."""
    dataset = evidence("dataset_shot_types")["metrics"]
    return tabular(
        "lll",
        ["\\emph{Dataset}", "Qué contiene", "Para qué"],
        [
            ["CVSPORTS~\\cite{decorte2024padel}",
             f"{count(dataset['rallies'])} puntos, {count(dataset['tournaments'])} torneos",
             "Detectar golpes y asignarlos"],
            ["Etiquetado propio", f"Tipo de sus {count(dataset['labelled'])} golpes",
             "Clasificar el tipo"],
            ["PadelTracker100~\\cite{badanerin2026padeltracker100}",
             "Pelota en dos finales", "Detectar la pelota"],
        ],
    )  # fmt: skip


def tools_table(lock: Path, web_lock: Path) -> str:
    """The libraries the system is built on, at the version the lock files pin."""
    locked = {p["name"]: p.get("version", "") for p in tomllib.loads(lock.read_text())["package"]}
    web = json.loads(web_lock.read_text())["packages"]
    python = tomllib.loads(lock.read_text())["requires-python"].strip("=.*")
    rows: list[list[str] | str] = [["Python", python, "Lenguaje de todo el sistema"]]
    for name, label, role, cite in [
        ("torch", "PyTorch", "Redes neuronales", "paszke2019pytorch"),
        ("torchaudio", "torchaudio", "Espectrograma log-Mel", "yang2022torchaudio"),
        ("ultralytics", "Ultralytics", "Pose (YOLO26) y seguimiento", "jocher2026yolo26"),
        ("opencv-python", "OpenCV", "Vídeo, dibujo y homografía", "bradski2000opencv"),
        ("numpy", "NumPy", "Cálculo numérico", "harris2020numpy"),
        ("matplotlib", "Matplotlib", "Figuras de la memoria", "hunter2007matplotlib"),
        ("pytest", "pytest", "Pruebas", ""),
        ("ruff", "Ruff", "Estilo del código", ""),
        ("mypy", "mypy", "Comprobación de tipos", ""),
    ]:
        name_cell = f"{label}~\\cite{{{cite}}}" if cite else label
        rows.append([name_cell, locked[name], role])
    for name, label, role in [
        ("react", "React", "Visor web"),
        ("vite", "Vite", "Construcción del visor"),
        ("motion", "Motion", "Animaciones del visor"),
    ]:
        rows.append([label, web[f"node_modules/{name}"]["version"], role])
    return tabular("lll", ["Herramienta", "Versión", "Para qué"], rows, rules=[10])

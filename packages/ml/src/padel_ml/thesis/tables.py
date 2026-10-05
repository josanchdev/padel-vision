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

DASH = "---"
CI = "IC 95\\,\\%"
RAGGED = ">{\\raggedright\\arraybackslash}p"
PAPER = "\\cite{decorte2024padel}"


def steps_table(evidence: Evidence) -> str:
    """Table 1: each step measured on its own, next to Decorte et al."""
    audio = evidence("audio_threshold_seeds")["metrics"]
    assignment = evidence("hit_assignment_replica")["metrics"]
    classifier = evidence("shot_type_classifier")["metrics"]
    player, player_low, player_high = evidence.interval("tabla1_asignacion.player")
    team, team_low, team_high = evidence.interval("tabla1_asignacion.team")
    accuracy, accuracy_low, accuracy_high = evidence.interval("tabla1_tipo.accuracy")
    return tabular(
        "llccc",
        ["Paso", "Medida", "Este trabajo", CI, f"Decorte et al.~{PAPER}"],
        [
            ["Cuándo", "Detección (F1)", proportion(audio["f1"]),
             f"$\\pm$\\,{proportion(audio['f1_std'])}", proportion(audio["paper_f1"], 2)],
            ["Quién", "Jugador", percent(player),
             percent_range(player_low, player_high), percent(assignment["paper_player_accuracy"])],
            ["", "Pareja", percent(team),
             percent_range(team_low, team_high), percent(assignment["paper_team_accuracy"])],
            ["Qué tipo", "\\emph{Accuracy}", percent(accuracy),
             percent_range(accuracy_low, accuracy_high), DASH],
            ["", "Macro-F1", proportion(classifier["macro_f1"]), DASH, DASH],
        ],
        rules=[1, 3],
    )  # fmt: skip


def system_table(evidence: Evidence) -> str:
    """Table 2: the whole chain, each tournament judged by models that never saw it."""
    rows = []
    for label, key, kind in [
        ("Detección de golpes (F1)", "detection_f1", "proportion"),
        ("Tipo de golpe (\\emph{accuracy})", "type_accuracy", "percent"),
        ("Jugador (VIGO)", "player_accuracy", "percent"),
        ("Pareja (VIGO)", "team_accuracy", "percent"),
        ("Golpe detectado y con el tipo correcto", "end_to_end_when_what", "percent"),
        ("Detectado, jugador y tipo correctos (VIGO)", "end_to_end_vigo", "percent"),
    ]:
        value, low, high = evidence.interval(f"tabla2_cadena.{key}")
        if kind == "proportion":
            rows.append([label, proportion(value), proportion_range(low, high)])
        else:
            rows.append([label, percent(value), percent_range(low, high)])
    return tabular("lcc", ["Medida", "Sistema completo", CI], rows, rules=[4])


def decisions_table(evidence: Evidence) -> str:
    """Each design decision against its alternative, measured on the same data."""
    same_data = evidence("method_comparison_same_data")["metrics"]
    classifier = evidence("shot_type_classifier")
    weights = classifier["params"]["class_weight_ablation"]
    windows = {
        (row["before"], row["after"]): row
        for row in evidence("assignment_window_sweep")
        if row["kind"] == "window" and row["offset"] == 0
    }
    ball = evidence("ball_detector")["params"]["by_tolerance"]
    raw, cleaned = ball["detector"]["15px"], ball["detector + limpieza"]["15px"]
    rows = [
        ["Detección: pose y pelota $\\to$ audio", "F1 (mismos datos)",
         proportion(same_data["old_f1"]), proportion(same_data["audio_f1"])],
        ["Regla del saque: sin $\\to$ con", "\\emph{Precision} del saque",
         proportion(classifier["params"]["serve_without_rule"]["precision"]),
         proportion(weights["0.0"]["serve_precision"])],
        ["", "Macro-F1", proportion(classifier["metrics"]["macro_f1_without_serve_rule"]),
         proportion(classifier["metrics"]["macro_f1"])],
        ["Pesos de clase: inversos $\\to$ sin pesos", "\\emph{Accuracy}",
         percent(weights["1.0"]["accuracy"]), percent(weights["0.0"]["accuracy"])],
        ["", "Macro-F1", proportion(weights["1.0"]["macro_f1"]),
         proportion(weights["0.0"]["macro_f1"])],
        ["Ventana de voto: $\\pm 6 \\to \\pm 4$ \\emph{frames}", "Jugador correcto",
         percent(windows[(6, 6)]["player"] / 100), percent(windows[(4, 4)]["player"] / 100)],
        ["Limpieza de la pelota: sin $\\to$ con", "\\emph{Recall} (15 px)",
         proportion(raw["recall"]), proportion(cleaned["recall"])],
        ["", "F1 (15 px)", proportion(raw["f1"]), proportion(cleaned["f1"])],
    ]  # fmt: skip
    return tabular(
        f"{RAGGED}{{0.36\\linewidth}}lcc",
        ["Decisión", "Medida", "Alternativa", "Elegida"],
        rows,
        rules=[1, 3, 5, 6],
    )


def ball_table(evidence: Evidence) -> str:
    """The ball detector at three distances, with and without the track cleaning."""
    ball = evidence("ball_detector")["params"]
    header = (
        "& \\multicolumn{3}{c}{Detector} & \\multicolumn{3}{c}{Detector y limpieza} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Tolerancia & F1 & \\emph{Precision} & \\emph{Recall} "
        "& F1 & \\emph{Precision} & \\emph{Recall} \\\\"
    )
    rows = []
    for tolerance in ball["tolerances_px"]:
        cells = [quantity(tolerance, 0, "px")]
        for variant in ("detector", "detector + limpieza"):
            scores = ball["by_tolerance"][variant][f"{tolerance}px"]
            cells += [proportion(scores[k]) for k in ("f1", "precision", "recall")]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(
        ["\\begin{tabular}{lcccccc}", "\\toprule", header, "\\midrule", *rows,
         "\\bottomrule", "\\end{tabular}"]
    )  # fmt: skip


def cost_table(evidence: Evidence) -> str:
    """Time per stage of `analyze_rally` on one rally."""
    metrics = evidence("processing_time")["metrics"]
    seconds = r"\second"
    return tabular(
        "lc",
        ["Etapa", "Tiempo"],
        [
            ["Audio: detector de golpes", quantity(metrics["audio_s"], 1, seconds)],
            ["Pose de los jugadores", quantity(metrics["pose_s"], 1, seconds)],
            ["Pelota", quantity(metrics["pelota_s"], 1, seconds)],
            ["Resto: vídeo, identidad, limpieza, voto y clasificador",
             quantity(metrics["resto_s"], 1, seconds)],
            ["Total", quantity(metrics["total_s"], 1, seconds)],
            ["Duración del vídeo", quantity(metrics["video_s"], 1, seconds)],
            ["Tiempo de proceso / duración del vídeo", proportion(metrics["realtime_factor"], 2)],
        ],
        rules=[4, 5],
    )  # fmt: skip


def datasets_table(evidence: Evidence) -> str:
    """The three sources of data and what each one is used for."""
    dataset = evidence("dataset_shot_types")["metrics"]
    hits = evidence("hit_assignment_replica")["metrics"]["hits"]
    return tabular(
        f"{RAGGED}{{0.24\\linewidth}}{RAGGED}{{0.36\\linewidth}}{RAGGED}{{0.28\\linewidth}}",
        ["\\emph{Dataset}", "Contenido", "Uso"],
        [
            ["CVSPORTS~\\cite{decorte2024padel}",
             f"{count(dataset['rallies'])} puntos de {count(dataset['tournaments'])} torneos "
             f"con el instante de {count(dataset['labelled'])} golpes; en {count(hits)} de "
             "ellos, quién golpea",
             "Entrenar y medir el detector de audio; medir la asignación"],
            ["Etiquetado propio",
             f"Tipo de esos {count(dataset['labelled'])} golpes: {count(dataset['trainable'])} de "
             f"cuatro clases y {count(dataset['discards'])} descartes",
             "Entrenar y medir el clasificador de tipo, por torneos"],
            ["PadelTracker100~\\cite{badanerin2026padeltracker100}",
             "Finales del WPT 2022 con la pelota anotada en cada \\emph{frame}",
             "Entrenar (final masculina) y medir (final femenina) el detector de pelota"],
        ],
    )  # fmt: skip


def tools_table(lock: Path, web_lock: Path) -> str:
    """The libraries the system is built on, at the version the lock files pin."""
    locked = {p["name"]: p.get("version", "") for p in tomllib.loads(lock.read_text())["package"]}
    web = json.loads(web_lock.read_text())["packages"]
    python = tomllib.loads(lock.read_text())["requires-python"].strip("=.*")
    rows = [["Python", python, "Lenguaje de todo el sistema"]]
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

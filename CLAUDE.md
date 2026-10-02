# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

**Padel Vision** — TFG de Ingeniería Informática (URJC): para cada golpe de un punto de pádel grabado con la cámara de retransmisión, el sistema dice cuándo (audio), quién (pose + pelota) y qué tipo de golpe es (BST-0 sobre pose + pelota). Parte de Novillo et al. (2024) *"Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings"*; el cuándo y el quién reimplementan Decorte et al. (CVPRW 2024), con quien se comparan los resultados. Ficha técnica: `docs/sistema.md`.

- El paper base vive como referencia en `~/reference/DS_Padel`. **NO tocar ni copiar código de ahí**; solo consulta conceptual.
- Diferencial: **clasificación automática del tipo de golpe**, con un dataset propio de 2.377 golpes etiquetados, y un visor web estático de puntos analizados.
- Este código se defiende ante un tribunal: prioriza código limpio, testeado y modular sobre soluciones rápidas y desechables.

## Decisiones de arquitectura (ADRs)

Todas las decisiones importantes se documentan en `docs/decisions/`. Antes de proponer una alternativa a algo ya decidido, consulta el ADR correspondiente y respeta la decisión salvo discusión explícita con Jorge. Si una decisión de diseño no está cubierta por un ADR existente, **propónla explícitamente en vez de decidir en silencio**.

Índice y estado de cada ADR en `docs/decisions/README.md`. Las que definen el sistema final: ADR-0015 (el audio detecta el golpe; la asignación replica el paper), ADR-0016 (clasificador BST sobre pose + pelota), ADR-0017 (web estática) y ADR-0018 (pista marcada a mano).

## Roadmap por niveles (histórico: así se construyó, "demo siempre verde")

Cada nivel debe quedar demostrable end-to-end y testeado antes de empezar el siguiente. **No adelantar trabajo de un nivel superior si el actual no está cerrado.**

- **Nivel 1 (MVP)**: pipeline court+player+pose detection, clasificador dummy de golpes, API básica, demo end-to-end en Docker Compose.
- **Nivel 2**: dataset propio anotado (CVAT), clasificador serio (ST-GCN u otra arquitectura, con comparación), dashboard pulido, métricas formales.
- **Nivel 3 (extensiones)**: detección de botes, trayectoria aproximada de pelota, modo live preview.
- **Nivel 4 (si todo va perfecto)**: recomendaciones tácticas, highlights.

**Estado actual (oct 2026): sistema cerrado.** Resultados (2 tablas, `docs/metrics/`), web estática y limpieza hechos; lo siguiente es la memoria en LaTeX (carpeta `memoria/`). Fuera de alcance, declarado en `docs/sistema.md`: marcador y resultado del punto, más clases de golpe, cámaras móviles. Historia experimental en `docs/experiments.md`.

## Stack técnico

- Monorepo con **uv workspace**: `packages/cv`, `packages/ml` (Python) y `packages/web` (React + Vite + Motion, estática)
- Python 3.11, PyTorch, OpenCV, YOLO26-pose, TrackNetV3, BST-0
- Evidencias en `docs/metrics/` (JSON + figuras, cada una regenerable con un script); MLflow solo registra el entrenamiento de la pelota
- Etiquetado con herramientas propias en OpenCV (`padel-cv annotate-types`, `annotate-court`)
- Linting: **ruff + mypy strict**

## Convenciones

- El sistema completo es una sola función, `padel_ml.rally_analysis.analyze_rally`; demo, web y evaluación usan ese mismo código
- Tests con pytest, en `tests/` dentro de cada package
- Commits en formato conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`)
- **Todo el código en inglés** (nombres, docstrings, comentarios); documentación de decisiones y memoria en español
- **Markdown para docs versionados** (ADRs, arquitectura, READMEs; diagramas con Mermaid). **HTML solo para producto generado** (informes de partido, dashboard, comparativas custom). Nunca HTML escrito a mano para docs del repo.
- ADRs en formato ligero (~15 líneas: Contexto / Decisión / Consecuencias); la documentación no debe frenar el desarrollo

## Cómo trabajar con Jorge

- Estudiante de 4º de Ingeniería Informática, con experiencia profesional en CV (detección/tracking con YOLO en producción, Azure ML, MLflow), pero **sin experiencia previa en**: action recognition basado en skeleton (ST-GCN y similares), multi-object tracking avanzado, arquitecturas de APIs asíncronas con colas.
- Al implementar algo de un área que no domina, explica brevemente el concepto antes o junto con el código.

## Comandos

```bash
uv sync --all-packages              # instalar todo el workspace
uv run pytest                       # tests (packages/*/tests)
uv run ruff check . && uv run mypy  # lint + tipos (strict)
uv run python scripts/demo_full_pipeline.py PUNTO.mp4 -o OUT.mp4   # un punto de principio a fin
uv run python scripts/export_points.py PUNTO.mp4 [...]              # puntos para la web
uv run padel-cv annotate-types | annotate-court VIDEO               # etiquetado propio
```

Datos en `data/` (gitignoreado salvo `labels/types/` y `datasets/courts/`): `raw/` vídeos, `labels/` PadelTracker100, `datasets/` cachés. Pesos entrenados en `runs/` (gitignoreado; no se publican). Entrenamiento y evidencias: ver `README.md` y `docs/metrics/README.md`.

**Aviso entorno**: el WSL de Jorge sufre crashes esporádicos ("catastrophic failure", causa sin diagnosticar, posiblemente bajo carga GPU/IO sostenida). Consolidar trabajo con commits frecuentes; entrenamientos largos con checkpoints reanudables (`resume=True`).

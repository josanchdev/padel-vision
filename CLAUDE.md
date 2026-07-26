# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

**Padel Vision** — TFG de Ingeniería de Computación (URJC): análisis automático de partidos de pádel con computer vision. Extiende el paper de Novillo et al. (2024) *"Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings"*.

- El paper base vive como referencia en `~/reference/DS_Padel`. **NO tocar ni copiar código de ahí**; solo consulta conceptual.
- Diferencial frente al paper original (que hace tracking 2D vía homografía): **clasificación automática del tipo de golpe a partir de pose corporal**, con plataforma web/API para el equipo de pádel de la URJC.
- Este código se defiende ante un tribunal: prioriza código limpio, testeado y modular sobre soluciones rápidas y desechables.

## Decisiones de arquitectura (ADRs)

Todas las decisiones importantes se documentan en `docs/decisions/`. Antes de proponer una alternativa a algo ya decidido, consulta el ADR correspondiente y respeta la decisión salvo discusión explícita con Jorge. Si una decisión de diseño no está cubierta por un ADR existente, **propónla explícitamente en vez de decidir en silencio**.

Decisiones cerradas:
- ADR-0001 — Detección de pista: keypoints aprendidos (no K-means/contornos)
- ADR-0002 — Alcance: pose para clasificar golpes, no solo tracking 2D
- ADR-0003 — Detector/pose: YOLO26-pose
- ADR-0004 — 2D vs 3D jugador: ABIERTO; empezar con 2D + suavizado temporal

## Roadmap por niveles (incremental, "demo siempre verde")

Cada nivel debe quedar demostrable end-to-end y testeado antes de empezar el siguiente. **No adelantar trabajo de un nivel superior si el actual no está cerrado.**

- **Nivel 1 (MVP)**: pipeline court+player+pose detection, clasificador dummy de golpes, API básica, demo end-to-end en Docker Compose.
- **Nivel 2**: dataset propio anotado (CVAT), clasificador serio (ST-GCN u otra arquitectura, con comparación), dashboard pulido, métricas formales.
- **Nivel 3 (extensiones)**: detección de botes, trayectoria aproximada de pelota, modo live preview.
- **Nivel 4 (si todo va perfecto)**: recomendaciones tácticas, highlights.

**Estado actual: Niveles 1, 2 y 3 cerrados** (pelota TrackNet V2/V3, botes, integración). El análisis de resultado (quién gana el punto, stats por jugador) NO es el siguiente paso: es la cima de una pila de robustez (cámara-en-pista, segmentación de puntos) documentada en `docs/backlog.md`; ese es el frente prioritario, a decidir en sesión futura. Historia experimental en `docs/experiments.md`.

## Stack técnico

- Monorepo con **uv workspace**: `packages/cv`, `packages/api`, `packages/web`
- Python 3.11, PyTorch, OpenCV, YOLO26-pose
- FastAPI + cola de jobs (a decidir: Celery/Redis vs Arq) para procesamiento batch
- MLflow para tracking de experimentos; CVAT para anotación
- Docker Compose para orquestación local
- Linting: **ruff + mypy strict**

## Convenciones

- Cada etapa del pipeline implementa la interfaz común `PipelineStage` (ver `docs/architecture.md` cuando exista)
- Tests con pytest, en `tests/` dentro de cada package
- Commits en formato conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`)
- **Todo el código en inglés** (nombres, docstrings, comentarios); documentación de decisiones y memoria en español
- **Markdown para docs versionados** (ADRs, arquitectura, READMEs; diagramas con Mermaid). **HTML solo para producto generado** (informes de partido, dashboard, comparativas custom). Nunca HTML escrito a mano para docs del repo.
- ADRs en formato ligero (~15 líneas: Contexto / Decisión / Consecuencias); la documentación no debe frenar el desarrollo

## Cómo trabajar con Jorge

- Estudiante de 4º de Ingeniería de Computación, con experiencia profesional en CV (detección/tracking con YOLO en producción, Azure ML, MLflow), pero **sin experiencia previa en**: action recognition basado en skeleton (ST-GCN y similares), multi-object tracking avanzado, arquitecturas de APIs asíncronas con colas.
- Al implementar algo de un área que no domina, explica brevemente el concepto antes o junto con el código.

## Comandos

```bash
uv sync --all-packages              # instalar todo el workspace
uv run pytest                       # tests (packages/*/tests)
uv run ruff check . && uv run mypy  # lint + tipos (strict)
uv run padel-cv process VIDEO -o OUT.mp4 [--court-model PESOS.pt | --homography GT.json]
uv run padel-cv sample-frames DIR -o OUT_DIR --per-video N
uv run padel-cv build-court-dataset VIDEO --homography GT.json -o DATASET_DIR --split train
tools/cvat/cvat.sh up|down          # entorno de anotación (ver tools/cvat/README.md)
```

Datos en `data/` (gitignoreado): `raw/` vídeos, `labels/` PadelTracker100, `datasets/` datasets generados. Pesos entrenados en `runs/` (gitignoreado).

**Aviso entorno**: el WSL de Jorge sufre crashes esporádicos ("catastrophic failure", causa sin diagnosticar, posiblemente bajo carga GPU/IO sostenida). Consolidar trabajo con commits frecuentes; entrenamientos largos con checkpoints reanudables (`resume=True`).

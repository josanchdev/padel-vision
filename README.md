# Padel Vision

Análisis automático de partidos de pádel con computer vision — TFG de Ingeniería de Computación (URJC).

Extiende el trabajo de Novillo et al. (2024), *Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings*, añadiendo clasificación automática del tipo de golpe a partir de pose corporal y una plataforma web/API de análisis.

## Estructura

- `packages/cv` — pipeline de visión (pista, jugadores, pose, golpes)
- `packages/api` — API FastAPI para procesamiento de vídeos
- `packages/web` — dashboard (placeholder)
- `docs/decisions/` — ADRs (decisiones de arquitectura)

## Desarrollo

```bash
uv sync --all-packages                     # instala todo el workspace
uv run pytest                              # tests
uv run ruff check . && uv run ruff format --check .   # lint
uv run mypy                                # type-check estricto
uv run padel-cv process data/raw/match.mp4 -o output/match_poses.mp4
```

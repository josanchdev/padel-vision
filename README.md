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

## Ejecución de la plataforma (Nivel 1)

**Stack completo en contenedores** (redis + API + worker GPU):

```bash
docker compose up -d          # requiere Docker Desktop con GPU (WSL2)
# → interfaz en http://localhost:8000 · API docs en http://localhost:8000/docs
docker compose down
```

**Modo desarrollo** (redis en Docker; API y worker nativos, iteración rápida):

```bash
./tools/dev.sh up             # api :8000 + worker con la GPU del host
./tools/dev.sh logs
./tools/dev.sh down
```

Flujo: subir un vídeo → se encola (Arq/Redis, ADR-0007) → el worker ejecuta el
pipeline de `packages/cv` → se descarga el vídeo anotado (pose, IDs J1-J4,
minimapa 2D, golpes). Detección de pista: modelo aprendido para vistas
arbitrarias o calibración exacta para cámaras fijas (ADR-0006).

# Padel Vision

Análisis automático de puntos de pádel a partir del vídeo de retransmisión. Para
cada golpe, el sistema responde:

- **cuándo** se golpea, con un detector de audio;
- **quién** golpea, con la pose de los jugadores y la trayectoria de la pelota;
- **qué tipo** de golpe es (derecha, revés, remate o saque), con un clasificador
  sobre la pose del jugador y la pelota.

Trabajo Fin de Grado de Ingeniería Informática, Universidad Rey Juan Carlos.
Parte del seguimiento 2D por homografía de Novillo et al. (2024). El cuándo y
el quién reimplementan el método de Decorte et al. (CVPRW 2024), con el que se
comparan los resultados. El tipo de golpe es la aportación propia, sobre un
dataset de 2.377 golpes etiquetados a mano.

**Ficha técnica y resultados:** [docs/sistema.md](docs/sistema.md).

## Estructura

| Carpeta | Contenido |
|---|---|
| `packages/cv` | Jugadores (YOLO26-pose e identidad J1-J4), geometría de pista y herramientas de etiquetado |
| `packages/ml` | Modelos y sistema: audio (CRNN), pelota (TrackNetV3), asignación, tipo de golpe (BST-0) y `analyze_rally`, el sistema completo |
| `packages/web` | Visor web estático de puntos analizados (React + Vite + Motion) |
| `scripts/` | Procesar puntos, entrenar los modelos y regenerar cada evidencia |
| `docs/` | Ficha técnica, [decisiones](docs/decisions/README.md), [evidencias](docs/metrics/README.md) y bitácora de experimentos |
| `data/` | Fuera de git salvo el etiquetado propio (`data/labels/types/`) y las pistas marcadas (`data/datasets/courts/`) |

## Instalación

Python 3.11 y [uv](https://docs.astral.sh/uv/). Entrenar o procesar vídeo pide
una GPU NVIDIA; la web, Node.js.

```bash
uv sync --all-packages
uv run pytest                       # tests
uv run ruff check . && uv run mypy  # lint y tipos (strict)
```

## Datos y modelos

Ni los vídeos ni los modelos entrenados se distribuyen. Datos de partida:

- **CVSPORTS_Padel** (Decorte et al.): 99 puntos con audio, el instante de cada
  golpe y, en el torneo de Vigo, quién golpea. Va en
  `data/raw/padel_audio_dataset/CVSPORTS_Padel/`.
- **PadelTracker100**: dos finales con la pelota anotada, para entrenar el
  detector de pelota. Va en `data/raw/` (vídeos) y `data/labels/` (anotaciones).

Los modelos se entrenan con:

```bash
# Cuándo: detector de audio
uv run python scripts/train_audio_detector.py

# Pelota: caché de frames de cada final y entrenamiento de TrackNetV3
uv run padel-cv extract-ball-frames data/raw/2022_BCN_FinalM_1.mp4 \
    --ball data/labels/2022_BCN_FinalM_1_ball.json -o data/datasets/ball_cache/finalM
uv run padel-cv extract-ball-frames data/raw/2022_BCN_FinalF_1.mp4 \
    --ball data/labels/2022_BCN_FinalF_1_ball.json -o data/datasets/ball_cache/finalF
uv run padel-ball-train --model tracknetv3 --epochs 40 --augment \
    --train-dir data/datasets/ball_cache/finalM --val-dir data/datasets/ball_cache/finalF \
    --out runs/ball_full/tracknetv3.pt --mlflow-uri sqlite:///runs/mlruns.db --resume

# Qué tipo: pose y pelota de los 99 puntos, y el clasificador
uv run python scripts/extract_rally_features.py
uv run python scripts/train_shot_classifier.py
```

## Uso

```bash
# Un punto de principio a fin: vídeo anotado (y, con --data, sus datos en JSON)
uv run python scripts/demo_full_pipeline.py PUNTO.mp4 -o anotado.mp4

# Puntos para la web, y la web
uv run python scripts/export_points.py PUNTO.mp4 [MÁS.mp4 ...]
cd packages/web && npm install && npm run dev    # http://localhost:5173
```

Si el torneo del vídeo aún no tiene pista marcada, se abre una ventana para
marcarla con 6 clics; se guarda una vez por torneo (ADR-0018).

## Etiquetado

```bash
uv run padel-cv annotate-types RALLY.mp4 -o data/labels/types/   # tipo de cada golpe
uv run padel-cv annotate-court RALLY.mp4                         # pista de un torneo
```

## Reproducir los resultados

Cada cifra de la memoria sale de un fichero de `docs/metrics/` y cada fichero,
de un script. La lista está en [docs/metrics/README.md](docs/metrics/README.md).

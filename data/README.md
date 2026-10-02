# Datos

Cada carpeta responde a una sola pregunta: **de terceros**, **propio** o
**regenerable**. Solo van a git este README y `labels/`, el etiquetado propio.
Las rutas que usa el código están en `packages/cv/src/padel_cv/paths.py`.

| Carpeta | Qué contiene | En git | Cómo se obtiene |
|---|---|---|---|
| `raw/cvsports_padel/` | **CVSPORTS_Padel** (Decorte et al., CVPRW 2024): 99 puntos de 11 torneos con audio (`rallies/`), el instante de cada golpe (`metadata/hits.csv`) y, en Vigo, quién golpea (`metadata/hit_assignments.xlsx`) | no | se descarga (abajo) |
| `raw/padeltracker100/` | **PadelTracker100** (Data in Brief, 2026, CC BY 4.0): las finales femenina y masculina de Barcelona 2022 con sus anotaciones y el artículo del dataset. Solo se usa la pelota (`*_ball.json`): la masculina para entrenar el detector y la femenina para medirlo | no | se descarga (abajo) |
| `labels/shot_types/` | **Etiquetado propio**: el tipo de cada golpe de los 99 puntos, un CSV por punto (`frame;type;from_wall`) | sí | `padel-cv annotate-types` |
| `labels/courts/` | **Pistas marcadas a mano**, una por torneo (6 clics): esquinas y homografía | sí | `padel-cv annotate-court` |
| `cache/rally_features/` | Pose, identidad J1-J4 y pelota de los 99 puntos | no | `scripts/extract_rally_features.py` (~1 h de GPU) |
| `cache/audio_features.npz` | Espectrogramas de los 99 puntos | no | se crea solo al entrenar el audio |
| `cache/ball_frames/` | Frames de PadelTracker100 preparados para entrenar la pelota | no | `padel-cv extract-ball-frames` (ver README raíz) |
| `processed_videos/` | Vídeos procesados durante el desarrollo, como archivo | no | — |

## Descarga

- **CVSPORTS_Padel**: https://github.com/robbedec/datasets/tree/master/CVsports/Padel
  → `raw/cvsports_padel/` (con `rallies/` y `metadata/` dentro).
- **PadelTracker100**: Zenodo, registro 17020011 → `raw/padeltracker100/`
  (`2022_BCN_FinalF_1.mp4`, `2022_BCN_FinalM_1.mp4` y sus `*_ball.json`).

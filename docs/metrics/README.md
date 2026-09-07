# Evidencias experimentales

Cada afirmación numérica de la memoria debe apoyarse en un fichero de esta
carpeta, no en una frase de la bitácora. Un número escrito en prosa no se puede
re-graficar, ni re-verificar, ni defender ante la pregunta "¿de dónde sale?".

`docs/metrics/` está **versionado en git** (a diferencia de `runs/`, que está
gitignoreado y vive solo en el disco de Jorge, que sufre crashes de WSL).

## Formato

Cada experimento es un JSON con la misma forma (`padel_ml.evidence.ExperimentResult`):

| Campo | Para qué sirve en la memoria |
|---|---|
| `summary` | La frase que se puede citar directamente |
| `metrics` | Los números de la tabla |
| `dataset` | Sobre QUÉ se midió (el split importa tanto como el número) |
| `method` | Qué enfoque lo produjo — permite tablas método-vs-método |
| `params` | Hiperparámetros: hace el resultado reproducible |
| `per_class` | Precisión/recall/F1 por clase (la media macro esconde las clases raras) |
| `confusion` + `labels` | Matriz de confusión como datos, re-graficable |
| `notes` | El PORQUÉ: por qué se hizo y qué se concluyó |
| `commit` | Commit exacto en que se midió |

## Cómo regenerar

```bash
uv run python scripts/evidence_audio_detector.py   # detector de audio + comparativa
uv run python scripts/evidence_hit_assignment.py   # asignación de golpe a jugador
```

Los scripts vuelven a medir desde cero y sobreescriben los JSON y las figuras de
`figures/`. Si un resultado no se puede regenerar con un script, no es evidencia:
es un recuerdo.

## Qué hay

| Fichero | Qué demuestra | Usado en |
|---|---|---|
| `audio_hit_detector.json` | Detector por audio F1 0,93 (paper 0,92) | ADR-0015 A |
| `audio_threshold_sweep.json` | Barrido de umbral (una ejecución) | ADR-0015 A |
| `audio_threshold_seeds.json` | **Elección del umbral**: F1/P/R por umbral, media de 3 ejecuciones | ADR-0015 A |
| `audio_training_loss.json` | Curva de entrenamiento del CRNN | ADR-0015 A |
| `hit_assignment_replica.json` | Asignación: equipo 86,83% = paper; matriz de confusión | ADR-0015 D |
| `hit_assignment_per_rally.json` | Tabla por rally, comparable con su Tabla 3 | ADR-0015 D |
| `shot_baseline_poseonly.json` | Baseline antiguo pose-sola (macro-F1 0,62) | ADR-0013 |
| `method_comparison_same_data.json` | **Cara a cara**: pose+pelota 0,821 vs audio 0,956, mismos datos | ADR-0015 |

### Figuras (`figures/`)

| Figura | Qué muestra |
|---|---|
| `audio_detector_training.png` | Curva de pérdida + sensibilidad al umbral |
| `audio_threshold_seeds.png` | Por qué 0,5: F1/precisión/recall por umbral con dispersión |
| `shot_detection_approaches.png` | **Por qué el audio**: recall de cada enfoque probado |
| `hit_assignment_confusion.png` | Matriz de confusión de la asignación |
| `hit_assignment_vs_paper.png` | Nuestra cifra frente a la publicada |
| `method_comparison_same_data.png` | **Por qué se cambió de método**, medido en igualdad de condiciones |

## Evidencias que NO se pueden regenerar (deuda documentada)

Honestidad metodológica: estos números están en la bitácora pero **no son
reproducibles hoy**, y la memoria debe presentarlos como tales.

- **Detector por ventana (29% recall)**: se midió sobre `citys_cup_1080.mp4`,
  vídeo que se perdió (se redescargó otra versión de YouTube que ya no alinea con
  las etiquetas). Las etiquetas (413 golpes) sí se conservan y siguen alineadas
  con la pose cacheada.
- ~~Localizer pose+pelota (61% recall)~~ → **DEUDA CERRADA**: re-medido sobre
  CVSPORTS en igualdad de condiciones (`method_comparison_same_data.json`),
  F1 0,821 frente a 0,956 del audio.
- **Heurística ADR-0012 (recall 15-52%)**: medida sobre PadelTracker100; es
  re-medible, pero el código de la heurística quedó superado.

Se conservan como parte de la narrativa del proyecto (el camino recorrido es
resultado en sí mismo), citando la bitácora como fuente y advirtiendo que fueron
mediciones puntuales, no evidencias regenerables.

## Fuera de esta carpeta

- `docs/experiments.md` — la bitácora: el relato cronológico y el porqué.
- `docs/decisions/` — los ADRs: qué se decidió y con qué argumento.
- `runs/` (gitignoreado) — pesos, curvas de Ultralytics y matrices de los
  entrenamientos de pista (`court_v*/results.csv`, `confusion_matrix.png`) y de
  pelota (`ball_plots_full/*.png`). **No están respaldados en git**: si hacen
  falta para la memoria, copiar las figuras relevantes a `figures/`.

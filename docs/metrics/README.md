# Evidencias experimentales

Cada afirmación numérica de la memoria se apoya en un fichero de esta carpeta, y
cada fichero lo escribe un script versionado. Un número escrito en prosa no se
puede re-graficar, ni re-verificar, ni defender ante la pregunta "¿de dónde sale?".

Todo se mide con la configuración exacta del sistema que se entrega
(`padel_ml.rally_analysis`): mismos modelos, pista marcada a mano, mismos
umbrales. Las cifras viven en los JSON; `docs/sistema.md` las cita.

## Formato

Cada experimento es un JSON con la misma forma (`padel_ml.evidence.ExperimentResult`):

| Campo | Para qué sirve en la memoria |
|---|---|
| `summary` | La frase que se puede citar directamente |
| `metrics` | Los números de la tabla |
| `dataset` | Sobre QUÉ se midió (el split importa tanto como el número) |
| `method` | Qué enfoque lo produjo |
| `params` | Configuración y resultados auxiliares: hace el resultado reproducible |
| `per_class` | Precisión/recall/F1 por clase (la media macro esconde las clases raras) |
| `confusion` + `labels` | Matriz de confusión como datos, re-graficable |
| `notes` | Por qué se hizo y cómo leerlo |
| `commit` | Commit en que se midió; con sufijo `-dirty` si había código sin commit |

## Cómo regenerar

Requieren los datos de `data/` y, salvo las estadísticas del dataset y las
evidencias de la cadena, GPU.

```bash
uv run python scripts/evidence_audio_detector.py       # cuándo: detector de audio, 3 ejecuciones
uv run python scripts/evidence_hit_assignment.py       # quién: réplica de la asignación del paper
uv run python scripts/experiment_assignment_window.py  # quién: ancho y posición de la ventana de voto
uv run python scripts/evidence_shot_classifier.py      # qué: por torneos, pesos de clase, regla del saque
uv run python scripts/evidence_dataset_stats.py        # el etiquetado propio
uv run python scripts/evidence_ball_detector.py        # pelota, tal como la usa el sistema
uv run python scripts/evaluate_chain_cv.py             # sistema completo por torneos (reentrena por ronda)
uv run python scripts/evidence_chain_cv.py             # evidencias y figuras del anterior
```

## Tabla 1: cada paso por separado

| Fichero | Qué demuestra | Script |
|---|---|---|
| `audio_threshold_seeds.json` | **Detección por audio**: F1 medio de 3 ejecuciones por umbral; es la cifra citada | `evidence_audio_detector.py` |
| `audio_hit_detector.json` | Detección por audio, una sola ejecución (semilla 0) | ídem |
| `audio_threshold_sweep.json` | Barrido de umbral de esa ejecución | ídem |
| `audio_training_loss.json` | Curva de entrenamiento del CRNN | ídem |
| `hit_assignment_replica.json` | **Asignación del golpe al jugador** sobre los 319 golpes del paper; matriz de confusión | `evidence_hit_assignment.py` |
| `hit_assignment_per_rally.json` | La misma, por rally, comparable con la tabla 3 del paper | ídem |
| `assignment_window_sweep.json` | Ancho, asimetría y desplazamiento de la ventana de voto | `experiment_assignment_window.py` |
| `shot_type_classifier.json` | **Tipo de golpe** dejando fuera cada torneo; pesos de clase y regla del saque | `evidence_shot_classifier.py` |
| `shot_type_per_tournament.json` | Acierto por torneo (11 rondas) | ídem |

## Tabla 2: el sistema completo

| Fichero | Qué demuestra | Script |
|---|---|---|
| `chain_cv.json` | **Cadena completa por torneos**: detección, jugador (VIGO), tipo y todo junto, con modelos que no vieron el torneo | `evaluate_chain_cv.py` + `evidence_chain_cv.py` |
| `chain_cv_hits.csv` | Cada golpe de CVSPORTS en la cadena completa | `evidence_chain_cv.py` |

## Componentes y datos

| Fichero | Qué demuestra | Script |
|---|---|---|
| `ball_detector.json` | **Detector de pelota** en un partido que no vio, a 15, 30 y 60 px, con y sin limpieza | `evidence_ball_detector.py` |
| `dataset_shot_types.json` | **Etiquetado propio**: 2.377 golpes, reparto y desbalance | `evidence_dataset_stats.py` |
| `dataset_per_tournament.json` | Golpes por torneo y clase (base del split por torneos) | ídem |

## Históricas

Medidas con métodos que el sistema ya no usa. Se conservan porque explican las
decisiones (por qué el audio y no la pose para detectar el golpe). Sus scripts
necesitaban código retirado: se regeneran desde la etiqueta git `pre-limpieza`.

| Fichero | Qué demuestra |
|---|---|
| `method_comparison_same_data.json` | Cara a cara en los mismos datos: detector por pose y pelota frente al audio |
| `method_comparison_sweep.json` | Barrido de umbral del detector por pose y pelota |
| `shot_baseline_poseonly.json` | Primer clasificador de tipo, solo con pose (julio, ADR-0013) |

## Figuras (`figures/`)

| Figura | Qué muestra |
|---|---|
| `audio_threshold_seeds.png` | Por qué 0,5: F1, precisión y recall por umbral, con la dispersión entre ejecuciones |
| `audio_detector_training.png` | Curva de pérdida y sensibilidad al umbral (una ejecución) |
| `shot_detection_approaches.png` | Por qué el audio: recall de cada enfoque probado |
| `method_comparison_same_data.png` | El cara a cara anterior, en igualdad de condiciones |
| `hit_assignment_confusion.png` | Matriz de confusión de la asignación |
| `hit_assignment_vs_paper.png` | Nuestra asignación frente a la publicada |
| `dataset_shot_types.png` | El dataset propio: clases y reparto por torneo |
| `shot_type_confusion.png` | Matriz de confusión del clasificador de tipo |
| `shot_classifier_training_curve.png` | Convergencia del clasificador: pérdida y acierto por época, 11 rondas |
| `chain_cv_funnel.png` | De cada 100 golpes reales: detectados, jugador correcto, tipo correcto (VIGO) |
| `chain_cv_per_tournament.png` | Sistema completo por torneo |
| `chain_cv_type_confusion.png` | Matriz de confusión del tipo en la cadena completa |

## Evidencias que no se pueden regenerar

Están en la bitácora (`docs/experiments.md`) pero no son reproducibles hoy, y la
memoria debe presentarlas como mediciones puntuales:

- **Detector por ventana (29% de recall)**: se midió sobre un vídeo de YouTube
  que ya no está disponible en la misma versión.
- **Heurística ADR-0012 (recall 15-52%)**: medida sobre PadelTracker100 con código
  retirado.

## Fuera de esta carpeta

- `docs/sistema.md`: la ficha técnica, que cita estas cifras.
- `docs/experiments.md`: la bitácora, con el relato cronológico y el porqué.
- `docs/decisions/`: los ADRs.
- `runs/` (gitignoreado): los modelos entrenados, las rondas de `chain_cv` y el
  registro MLflow del entrenamiento de la pelota. No se publican.

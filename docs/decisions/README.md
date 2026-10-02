# Decisiones de arquitectura (ADRs)

Cada decisión importante, con su contexto y sus consecuencias. Las superadas se
conservan: el camino recorrido también explica el sistema final.

| ADR | Decisión | Estado |
|---|---|---|
| [0001](0001-court-detection-learned-keypoints.md) | Detección de pista con keypoints aprendidos | superada por 0018 |
| [0002](0002-scope-pose-shot-classification.md) | Alcance: clasificar el tipo de golpe a partir de la pose | vigente |
| [0003](0003-detector-yolo26-pose.md) | Personas y pose con YOLO26-pose | vigente |
| [0004](0004-player-2d-vs-3d.md) | Jugador en 2D o 3D | cerrada en 2D |
| [0005](0005-generalization-over-benchmark.md) | Generalizar a cualquier vídeo, no al benchmark | vigente |
| [0006](0006-court-two-track.md) | Pista en dos vías: detector y calibración estática | superada por 0018 |
| [0007](0007-job-queue-arq.md) | Cola de trabajos Arq sobre Redis | superada por 0017 |
| [0008](0008-shot-classifier-design.md) | Clasificador de golpes ST-GCN / PoseConv3D | superada por 0016 |
| [0009](0009-ball-detection-tracknet.md) | Pelota con TrackNet y botes | vigente la pelota; botes retirados |
| [0010](0010-structured-data-layer.md) | Datos estructurados como producto | vigente, con `point.json` como único formato |
| [0011](0011-web-stack-react-vite-motion.md) | Web con React, Vite y Motion | vigente; el despliegue lo cambia 0017 |
| [0012](0012-shot-detection-by-ball.md) | Detectar golpes por la pelota | superada por 0013 |
| [0013](0013-shot-two-stage-learned.md) | Golpes con dos modelos aprendidos | superada por 0015 |
| [0014](0014-shot-labeling-quick-mark.md) | Etiquetado por marca rápida | superada en parte |
| [0015](0015-shot-audio-detect-rgb-classify.md) | El audio detecta el golpe; la asignación replica el paper | vigente |
| [0016](0016-shot-type-classifier.md) | Clasificador de tipo BST sobre pose y pelota | vigente |
| [0017](0017-static-web-viewer.md) | La web es un visor estático | vigente |
| [0018](0018-court-manual-only.md) | La pista se marca a mano | vigente |

Formato ligero: Contexto / Decisión / Consecuencias, unas 15 líneas.

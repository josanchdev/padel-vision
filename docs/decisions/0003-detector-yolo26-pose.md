# ADR-0003: Detector de personas y pose — YOLO26-pose

**Estado:** aceptada · 2026-07-12

## Contexto

Necesitamos detección de jugadores + estimación de pose (17 keypoints COCO) por frame, con buen rendimiento en vídeo. Alternativas consideradas: YOLO26-pose (Ultralytics), YOLO11-pose, RTMPose/MMPose (top-down), ViTPose.

## Decisión

YOLO26-pose (Ultralytics): detección + pose en una sola pasada (bottom-up sobre detección integrada), API madura con tracking integrado (ByteTrack/BoT-SORT), export sencillo y rendimiento estado del arte en 2026. El equipo ya tiene experiencia productiva con el ecosistema YOLO.

## Consecuencias

- (+) Una sola inferencia da cajas + esqueletos; simplifica el pipeline y el presupuesto de cómputo.
- (+) Tracking multi-objeto disponible de serie para asignar IDs estables a los 4 jugadores.
- (−) Acoplamiento al ecosistema Ultralytics (licencia AGPL-3.0: aceptable para un TFG académico, revisar si hubiera explotación comercial).
- (−) Precisión de keypoints inferior a métodos top-down dedicados (RTMPose) en poses extremas; si el clasificador de golpes sufre por calidad de pose, reevaluar en un ADR nuevo.

# ADR-0005: Generalización a cualquier vídeo como requisito de diseño

**Estado:** aceptada · 2026-07-12

## Contexto

Disponemos de PadelTracker100 (NTT Data, CC BY 4.0): 2 partidos WPT 2022 anotados con pelota, pose, golpes y homografía. Sus anotaciones se generaron con procesos manuales por vídeo (selección de planos a mano, homografía calibrada clicando puntos, refinado manual). Existe la tentación de construir el sistema "contra" este dataset y que solo funcione en esos vídeos.

## Decisión

El sistema debe funcionar sobre vídeos arbitrarios de pádel (caso objetivo: el equipo de la URJC coloca una cámara elevada cualquiera, sin calibración manual). PadelTracker100 se usa como material de entrenamiento y benchmark, nunca como supuesto de diseño:

- Prohibido cualquier paso de calibración manual por vídeo en el pipeline de producción.
- El detector de keypoints de pista (ADR-0001) se entrena con frames de fuentes diversas (múltiples torneos, ángulos, iluminación, pista URJC), no solo de PadelTracker100.
- El clasificador de golpes usa esqueletos normalizados (invarianza parcial a la vista) y se evalúa con test cruzado de dominio: entrenar en WPT, testear en grabaciones URJC.
- Ante una vista insuficiente (pocos keypoints de pista detectados), el sistema degrada con aviso y confianza baja, no falla en silencio.

## Consecuencias

- (+) Diferenciación clara frente a NTT Data (dataset manual) y frente al paper base: el TFG entrega la automatización de los pasos que ellos hicieron a mano.
- (+) La evaluación cruzada de dominio (pro→amateur, cámara fija→cámara libre) se convierte en experimento central de la memoria.
- (−) Exige anotar un dataset propio de keypoints de pista con diversidad de vistas (coste bajo: cientos de frames).
- (−) Límite físico documentado: vistas a ras de pista no son recuperables; el requisito es "cualquier vista elevada razonable".

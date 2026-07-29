# ADR-0013: Golpes como dos modelos aprendidos (detector → clasificador)

**Estado:** aceptada · 2026-07-29 · **supera a ADR-0012**

## Contexto

La detección de golpes ha pasado por dos heurísticas: pico de velocidad de muñeca
(ADR-0008) y cambio de dirección de la pelota + muñeca cerca (ADR-0012). La
validación métrica de ADR-0012 contra el GT de PadelTracker100 (`padel_ml.shot_eval`,
±3 frames al impacto, con pose+pelota GT para aislar el algoritmo) es concluyente:

| Partido | Golpes reales | Recall | Precisión |
|---|---|---|---|
| FinalM | 466 | **15 %** | 16 % |
| FinalF | 440 | **52 %** | 31 % |

Falla con pose y pelota **perfectas** → el problema es el **método**, no los
detectores. Causas: (1) el impacto ocurre cuando la pelota va más rápida y se
ocluye, así que a menudo no hay trayectoria que medir; (2) el "cambio de
dirección" no coincide de forma fiable con el contacto (timing malo, la queja del
partido real). Una regla geométrica fija es demasiado rígida para algo tan variable.

**Hallazgo del estado del arte:** BST (CVPR 2026) y TemPose (CVPR 2023) NO
detectan el golpe — parten de clips ya recortados en torno al contacto (el
dataset lo anota). El SotA **separa localizar el golpe de clasificar su tipo**.
Nosotros los mezclábamos en una heurística que hacía mal las dos cosas.

**Insight de Jorge:** un humano no necesita esqueleto ni reglas para saber que
alguien ha golpeado — lo aprende viendo partidos. La arquitectura correcta es la
misma que usábamos en detección industrial (Ferrovial): un **detector** genérico
alimenta a un **clasificador** especializado (detector de camiones → clasificador
de tipo de camión). Aplicado aquí: Modelo 1 detecta el golpe, Modelo 2 lo tipifica.

## Decisiones

- **A — Two-stage aprendido.** Se sustituye la heurística por dos modelos:
  - **Modelo 1 (detector de golpe):** dada una ventana temporal de pose+pelota,
    predice "¿hay golpe en este frame?" (y de quién). Aprendido, no una regla.
  - **Modelo 2 (clasificador de tipo):** por cada golpe localizado, un clip
    centrado en el contacto → tipo (derecha/revés/remate/saque/…).
  Los dos problemas se atacan por separado, como en el SotA y como en Ferrovial.
- **B — Entrada = pose + pelota** (no vídeo crudo). El detector aprende sobre las
  señales ya extraídas (esqueletos YOLO26 + trayectoria TrackNet). Reusa lo ya
  construido, es lo que valida la literatura de deportes de raqueta, y es un
  modelo ligero entrenable con nuestro GT. El vídeo crudo (fin al estilo "como el
  humano ve") queda como extensión futura documentada, no como el enfoque base.
- **C — Objetivo de recall del detector: ~85-95%** (rango del SotA), medido con
  `padel_ml.shot_eval` sobre PadelTracker100. Sustituye el 15-52% de la heurística.
- **D — El clasificador evoluciona a pose+pelota** (estilo TemPose/BST) frente al
  PoseConv3D solo-pose (0,63). Diseño de la fusión en ADR aparte cuando se aborde
  el Modelo 2; la contribución científica es la comparativa pose-sola vs pose+pelota.
- **E — GT de entrenamiento:** los bloques `has_shot` de PadelTracker100 (906
  golpes entre FinalM+FinalF) etiquetan el detector; su `category` etiqueta el
  clasificador. Sigue siendo material de entrenamiento, no dependencia de
  inferencia (ADR-0005): en producción todo sale de nuestros modelos.

## Consecuencias

- (+) Ataca la causa raíz medida: la localización deja de ser una regla frágil y
  pasa a un modelo con techo de recall del SotA.
- (+) Arquitectura limpia y defendible (dos contribuciones separadas, patrón
  detector→clasificador conocido); alineada con el estado del arte 2026.
- (+) Reusa pose+pelota ya construidas; el trabajo de la pelota era el ingrediente
  que faltaba, ahora justificado doblemente (detectar Y clasificar).
- (−) Dos modelos que entrenar, versionar y evaluar (más que una heurística).
- (−) Requiere diseñar la arquitectura temporal del detector (frente nuevo:
  action localization); se aborda en sesión dedicada con su propio ADR de diseño.
- (−) La heurística ADR-0012 se retira del camino de producción; se conserva en
  el código como baseline comparativo (heurística vs aprendido) para la memoria.

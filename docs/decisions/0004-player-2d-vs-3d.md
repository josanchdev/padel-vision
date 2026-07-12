# ADR-0004: Representación del jugador — 2D vs 3D

**Estado:** ABIERTA (decisión provisional: 2D) · 2026-07-12

## Contexto

Para clasificar golpes a partir de pose hay dos opciones: trabajar con keypoints 2D en coordenadas de imagen, o estimar pose 3D (lifting monocular). La pose 3D es más informativa e invariante al punto de vista, pero añade complejidad, error acumulado y coste de cómputo, todo desde una sola cámara.

## Decisión provisional

Empezar con pose 2D + suavizado temporal (filtrado de jitter entre frames) y normalización (centrado en cadera, escala por tamaño de torso) para reducir la dependencia del punto de vista. Reevaluar cuando existan métricas del clasificador de Nivel 2: si la precisión se estanca por ambigüedad de vista, probar lifting 3D como experimento comparativo.

## Consecuencias

- (+) Pipeline más simple y rápido; menos fuentes de error para el MVP.
- (+) La comparación 2D vs 3D, si se hace, es en sí un experimento valioso para la memoria.
- (−) La pose 2D depende del ángulo de cámara; mitigado parcialmente con normalización.

# ADR-0001: Detección de pista mediante keypoints aprendidos

**Estado:** aceptada · 2026-07-12

## Contexto

El paper base (Novillo et al., 2024) detecta la pista con métodos clásicos (segmentación por color/contornos) y reconoce en sus conclusiones que falla con vistas parciales de pista y que un dataset anotado de pistas mejoraría el sistema. Necesitamos las esquinas/intersecciones de la pista para calcular la homografía que proyecta posiciones de imagen a coordenadas reales.

## Decisión

Entrenar un modelo de keypoints (regresión de puntos característicos de la pista: esquinas, intersecciones de líneas, base de la red) en lugar de K-means/contornos. La anotación de keypoints de pista es barata (pocos clics por frame, pocos cientos de frames).

## Consecuencias

- (+) Robusto a variaciones de color de pista, iluminación y oclusiones parciales; funciona con pista parcialmente visible si se detectan suficientes puntos (≥4 para homografía).
- (+) Ataca directamente la primera línea de "future work" del paper base.
- (−) Requiere anotar un pequeño dataset de keypoints de pista antes de poder entrenar.

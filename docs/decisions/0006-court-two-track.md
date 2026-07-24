# ADR-0006: Detección de pista en dos vías — detector aprendido + calibración estática

**Estado:** aceptada · 2026-07-17

## Contexto

Seis entrenamientos (bitácora en `docs/experiments.md`) demostraron que propagar masivamente la anotación de cámaras fijas (500 frames casi-duplicados) envenena el detector: enseña a memorizar el layout en vez de localizarlo (sesgo de "pista encogida" medido en WPT: 0,22→0,70 m) y desestabiliza el entrenamiento. A la vez, medimos que las cámaras de trípode tienen deriva <1 px en 73 min: para ellas, la homografía calculada una vez de la anotación es exacta por definición — un detector solo puede empeorarla.

## Decisión

Dos vías según el tipo de cámara:

- **Cámara fija (URJC, PADELVIC)**: `StaticCourtStage` — anotar los 13 puntos una vez (CVAT, ~1 min), homografía exacta constante. Sin modelo, sin coste por frame.
- **Vista arbitraria o móvil (broadcast, vídeos desconocidos)**: `CourtDetectionStage` con el detector entrenado en receta estable (1280/batch16) sobre WPT + anotaciones manuales diversas, **sin propagación**. La generalización a pistas nuevas se compra con diversidad real (lote YouTube, Nivel 2), no con duplicados.

Los frames propagados de PADELVIC quedan como set de evaluación.

## Consecuencias

- (+) Cada vía usa la herramienta óptima; el caso URJC queda resuelto con precisión exacta y coste cero.
- (+) Desbloquea el cierre del Nivel 1 sin una séptima iteración de entrenamiento a ciegas.
- (−) Un vídeo de cámara fija sin anotación previa cae en la vía del detector (funciona, con menos precisión) hasta que alguien lo anote.
- (−) La vía estática asume cámara realmente fija; una cámara golpeada requiere re-anotar (detectable: la homografía deja de casar con las líneas).

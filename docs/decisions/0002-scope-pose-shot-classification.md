# ADR-0002: Alcance — clasificación de golpes basada en pose

**Estado:** aceptada · 2026-07-12

## Contexto

El paper base hace tracking 2D de jugadores y pelota. Sus conclusiones señalan que "incorporating information about the types of shots made by players will add significant detail" como trabajo futuro. Hay que elegir el diferencial del TFG: mejorar el tracking existente o añadir una capacidad nueva.

## Decisión

El diferencial principal del TFG es la clasificación automática del tipo de golpe a partir de la pose corporal (skeleton-based action recognition), no la mejora incremental del tracking. El tracking 2D se implementa con herramientas modernas como base necesaria, pero la contribución es la capa de análisis técnico de golpes.

## Consecuencias

- (+) Contribución clara y diferenciada frente al trabajo de los tutores; implementa su propio "future work".
- (+) Área poco explorada en pádel (vs tenis), con recorrido académico (comparación de arquitecturas, métricas propias).
- (−) Exige un dataset propio anotado de golpes (Nivel 2) — es el camino crítico del proyecto.
- (−) Requiere resolver también la segmentación temporal (cuándo ocurre un golpe), no solo la clasificación (pendiente de ADR propio antes del Nivel 2).

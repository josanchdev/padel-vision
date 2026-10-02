# ADR-0018: La pista se marca a mano, una vez por torneo

**Estado:** aceptada · 2026-09-30 · **supera ADR-0001 y ADR-0006**, y la
Decisión D2 de ADR-0015 (manual + detector automático de respaldo)

## Contexto

Había dos vías para la homografía: el detector aprendido de keypoints (court
v6, ADR-0001/0006) y el marcado manual de 6 puntos por torneo (ADR-0015 D2).
Medido sobre los 11 torneos de CVSPORTS, el detector reproyecta a 0,11-0,15 m
en seis, deriva a 1-5 m en tres y no encuentra la pista en dos. Una pista
equivocada no avisa: estropea en silencio todo lo que depende de ella (la
máscara que separa jugadores de público, las posiciones en metros, los mapas
de calor). El marcado manual cuesta unos segundos por torneo, porque la cámara
de retransmisión es fija, y es el mismo método que usa el paper de referencia.

## Decisión

- La pista se marca solo a mano: `padel-cv annotate-court`, 6 clics, un JSON
  por torneo en `data/datasets/courts/` (versionado).
- Al procesar un vídeo sin pista marcada, se abre la herramienta de marcado y
  el proceso continúa con la pista recién guardada.
- Se elimina el detector automático: su código, su modelo, sus datasets y la
  herramienta de anotación CVAT con la que se prepararon. Quedan en la etiqueta
  git `pre-limpieza`.

## Consecuencias

- Cada torneo nuevo requiere marcar su pista una vez; los demás puntos del
  mismo torneo la reutilizan.
- Detectar la pista no es objeto del TFG: la aportación está en el golpe.
- Todas las cifras se miden con pistas marcadas a mano, incluida la asignación
  de la tabla 1, que antes usaba la homografía automática de VIGO.

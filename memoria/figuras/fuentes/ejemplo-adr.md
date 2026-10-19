# ADR-0018: La pista se marca a mano, una vez por torneo

**Estado:** aceptada · 2026-09-30 · supera a ADR-0001 y ADR-0006

## Contexto

Para pasar de píxeles a metros hace falta la homografía de la pista, y había dos
formas de obtenerla: un detector aprendido de puntos clave (ADR-0001) o el
marcado manual de seis puntos por torneo. Medido sobre los 11 torneos de
CVSPORTS, el detector sitúa la pista con un error de 0,11 a 0,15 m en seis
torneos, se desvía de 1 a 5 m en tres y no la encuentra en dos.

Una pista equivocada no avisa: estropea en silencio todo lo que depende de ella,
como la máscara que separa a los jugadores del público, las posiciones en metros
o los mapas de calor. En cambio, marcarla a mano cuesta unos segundos por
torneo, porque la cámara es fija.

## Decisión

- La pista se marca solo a mano, con seis clics: las cuatro esquinas y los
  extremos de la red.
- Las marcas se guardan en un fichero por torneo y las reutilizan todos sus
  puntos.
- Si un vídeo no tiene pista marcada, se abre la herramienta de marcado antes de
  procesarlo.
- Se elimina el detector automático; su código queda en la etiqueta git
  `pre-limpieza`.

## Consecuencias

- Cada torneo nuevo exige marcar su pista una vez.
- La detección de la pista deja de ser objeto del trabajo: la aportación está en
  el golpe.
- Todas las cifras se miden con pistas marcadas a mano.

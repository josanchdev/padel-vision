# ADR-0015: Golpes = AUDIO detecta + RGB clasifica (supera ADR-0013)

**Estado:** aceptada · 2026-02 · **supera a ADR-0013** (dos modelos aprendidos sobre pose+pelota)

## Contexto

Toda la línea de detección de golpes por POSE+PELOTA se agotó, medido paso a paso
(docs/experiments.md): heurística ADR-0012 (recall 15-52%), detector aprendido
por-ventana (meseta en rallies), localizer por-frame (61% recall / 70% precisión,
insuficiente). El paso atrás + investigación del estado del arte
(docs/research-shot-detection-2026.md, docs/bibliography.md) dio dos hallazgos
decisivos:

1. **El problema es Precise Event Spotting**, y reducir a pose+pelota TIRA
   información (la pala, el gesto fino, el contexto) que el SotA conserva usando
   señales ricas (audio, RGB).
2. **Paper de PÁDEL (Decorte et al., CVPRW 2024): F1 92 % detectando golpes por
   AUDIO** con un CRNN sobre log-Mel. El golpe SUENA; es binario (hit/no-hit). Y
   publican dataset abierto con audio + 2377 golpes anotados (CVSPORTS_Padel).

Insight de Jorge: el audio detecta CUÁNDO (barato, alto recall); el RGB —como un
humano viendo la jugada— clasifica QUÉ tipo y puede además VERIFICAR (rechazar
falsos del audio). Cada señal en lo que es mejor.

## Decisiones

- **A — Detector = AUDIO (CRNN SED), binario.** Se replica el método del paper de
  pádel (log-Mel 40 bins → 3×conv2D + 2×GRU bidireccional → per-frame sigmoid,
  binary focal cross-entropy). El audio se extrae del PROPIO vídeo (móvil/YouTube)
  con ffmpeg — no un fichero aparte. Reemplaza todo lo de pose+pelota para
  DETECTAR. Se replica primero (baseline con F1 conocido), mejoras después si hay
  margen.
- **B — Clasificador de TIPO = RGB (contribución propia).** El paper para en la
  detección binaria; clasificar el tipo (derecha/revés/remate/saque/…) es NUESTRA
  aportación. Se hace sobre RGB (píxeles) porque conserva la información que
  pose+pelota tira. El RGB además puede emitir "no-golpe" → hace de VERIFICADOR
  del audio (dos señales independientes, oído+vista, se confirman).
- **C — Diseño fino del RGB: PENDIENTE de investigar.** Frame entero (contexto,
  no necesita saber "quién") vs recorte del golpeador (detalle del gesto, necesita
  "quién"); ventana del golpe entero (backswing→impacto→follow-through) vs solo
  impacto. NO se decide inventando: se leerá el/los papers de RGB/PES a fondo
  (como se hizo con el audio) cuando el audio esté validado. Hipótesis de partida:
  golpe entero (el gesto está en el movimiento) y probablemente contexto + detalle.
- **D — La PELOTA se mantiene** como señal de apoyo (trayectoria, "quién golpea"
  por proximidad), NO como detector de golpe (eso falla). TrackNet ya funciona
  (F1 0,94). El paper también usa TrackNet.
- **E — Orden:** (1) construir y validar el audio sobre su dataset (F1 + velocidad);
  (2) con el audio funcionando, investigar RGB a fondo y cerrar su diseño; (3) ADR
  de ampliación con el diseño RGB definitivo.
- **F — Datos: usar lo que sirva, citar todo, descartar lo que no aporte.**
  CVSPORTS_Padel (audio+hits) para el detector; PadelTracker100 + etiquetado propio
  (tipo) para el clasificador; datasets que dejen de aportar se descartan sin drama.

## Consecuencias

- (+) Cada señal en su fuerte: audio (cuándo, barato, F1 92 % demostrado en pádel),
  RGB (qué tipo, conserva info). El audio filtra los momentos → el RGB (caro) solo
  corre en candidatos, no en todo el vídeo → eficiente.
- (+) Verificación cruzada audio↔RGB = robustez (rechaza falsos del audio).
- (+) El TIPO por RGB es contribución científica (el paper no lo hace).
- (+) Se replica un método publicado (defendible, citable) en vez de inventar.
- (−) Dos modalidades nuevas (audio, RGB) que montar; RGB es pesado y quizá exige
  muchos datos de tipo (a evaluar).
- (−) El "quién golpea" sigue siendo un punto a resolver (pelota/pose, o que el RGB
  del frame entero lo aprenda) — parte del diseño fino pendiente (Decisión C).
- (−) Se retira pose+pelota como DETECTOR; el código del localizer/detector se
  conserva como baseline comparativo para la memoria (heurística vs pose+pelota vs
  audio).

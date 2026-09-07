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

- **A — Detector = AUDIO (CRNN SED), binario. ✅ HECHO (F1 0,93).** Replicado el
  método del paper (log-Mel 40 bins → 3×conv2D pool-frecuencia + 2×GRU bi →
  per-frame sigmoid, binary focal cross-entropy). Entrenado en CVSPORTS_Padel,
  split cross-rally, eval event-based (collar 250 ms): **F1 0,928 · prec 0,99 ·
  recall 0,87** — reproduce el paper (0,92). El audio se extrae del PROPIO vídeo
  (móvil/YouTube) con ffmpeg. Módulos: audio_dataset/audio_detector/audio_train.
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
- **D — El "QUIÉN golpea" se REPLICA del paper (no versión ingenua).** Decisión de
  Jorge: esto es el BACKBONE, y hacerlo ingenuo (1 frame muñeca-pelota) es
  inaceptable — cada eslabón multiplica al siguiente (detectar 30% × clasificar 25%
  = nefasto; detectar 93% × jugador 84% = base sólida). Se replica el método del
  paper (accuracy 83,7% jugador / 86,8% equipo):
  - **Asignación por voto ponderado multi-frame** (ventana 500 ms/12 frames, peso
    por distancia euclídea estandarizada, ec. 1), con fallbacks en cascada
    pose→muñecas→bbox→bbox promedio ±2 s, + barrido secundario que usa la
    alternancia de equipos.
  - **Re-identificación de tracks por aparición/desaparición** (no visual — los
    jugadores del mismo equipo visten igual): número de jugadores conocido (4),
    lógica de "si un track termina para J1 y los demás siguen, el nuevo es J1".
    Más elegante que nuestro ByteTrack+anclaje geométrico → se adopta el suyo.
  - La PELOTA (TrackNet, F1 0,94) es la señal de "quién" por proximidad, como en el
    paper. Se mantiene; NO como detector de golpe (eso falla).
- **D2 — Homografía/pista: manual + fallback automático. ✅ El automático GANA.**
  El paper usa puntos manuales por torneo (cámara fija) tras descartar su método
  automático por color. Se mantiene la opción manual, pero medido en VIGO nuestro
  detector v6 agregando 136 detecciones (mediana por keypoint; la cámara es fija,
  MAD 2-14 px) da **0,113 m de error** sin intervención manual — mejor que nuestro
  propio benchmark (0,196 m). **Mejora sobre el paper**, no solo réplica.
- **D3 — El detector emite VENTANAS, no picos (opción B).** El paper usa los
  límites onset/offset que predice el modelo y solo rellena hasta 500 ms si la
  ventana es corta. Se replica así (`windows_from_frames`): es más fiel y la
  anchura de la ventana es información aprovechable por el clasificador de tipo
  (un slice no suena como un remate).
- **E — Orden:** (1) detector de audio ✅ HECHO (F1 0,93); (2) replicar el backbone
  de asignación (quién golpea) + re-id + homografía manual del paper; (3) validar el
  audio en vídeo propio (generalización); (4) investigar RGB a fondo y diseñar el
  clasificador de TIPO (contribución); (5) ADR de ampliación con el diseño RGB.
- **F — Datos: usar lo que sirva, citar todo, descartar lo que no aporte.**
  CVSPORTS_Padel (audio+hits) para el detector; PadelTracker100 + etiquetado propio
  (tipo) para el clasificador; datasets que dejen de aportar se descartan sin drama.
- **G — Reuso del paper: reimplementar + citar (legítimo).** Los métodos descritos
  (asignación, re-id, homografía) se REIMPLEMENTAN a partir de su descripción
  citando el paper — es lo estándar y correcto en investigación (las ideas no se
  "copian", se implementan). El dataset es abierto → usar y citar. Nuestra
  contribución ORIGINAL es el clasificador de TIPO, que el paper no hace.

## El backbone (mapa completo)

```
Vídeo (con audio)
  ├─ AUDIO → CRNN SED → "hit en t"          ✅ F1 0,93 (replicado)
  ├─ POSE (YOLO) + re-id aparición/desap.   → 4 jugadores estables (replicar paper)
  ├─ PELOTA (TrackNet) + homografía         → posición en pista
  └─ hit + pelota + pose → voto multi-frame → QUIÉN golpeó (replicar paper, ~84%)
        ↓  (hit localizado + jugador conocido)
  RGB del golpeador → CLASIFICADOR DE TIPO  ← NUESTRA CONTRIBUCIÓN (a diseñar)
        → derecha/revés/remate/saque  (+ "no-golpe" = verificador del audio)
```
Calidad del backbone = techo del sistema. Por eso se replica el SotA (no versiones
ingenuas) antes de añadir el clasificador.

## Consecuencias

- (+) Cada señal en su fuerte: audio (cuándo, barato, F1 92 % demostrado en pádel),
  RGB (qué tipo, conserva info). El audio filtra los momentos → el RGB (caro) solo
  corre en candidatos, no en todo el vídeo → eficiente.
- (+) Verificación cruzada audio↔RGB = robustez (rechaza falsos del audio).
- (+) El TIPO por RGB es contribución científica (el paper no lo hace).
- (+) Se replica un método publicado (defendible, citable) en vez de inventar.
- (−) Dos modalidades nuevas (audio, RGB) que montar; RGB es pesado y quizá exige
  muchos datos de tipo (a evaluar).
- (+) El "quién golpea" tiene receta probada del paper (voto multi-frame + re-id +
  alternancia, ~84%) → se replica, no se inventa (Decisión D).
- (−) Se retira pose+pelota como DETECTOR; el código del localizer/detector se
  conserva como baseline comparativo para la memoria (heurística vs pose+pelota vs
  audio).

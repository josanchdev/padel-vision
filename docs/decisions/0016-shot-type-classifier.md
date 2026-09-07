# ADR-0016: Clasificador de TIPO de golpe = pose+pelota (BST), 4 clases con rechazo

**Estado:** aceptada · 2026-09-07 · **corrige la Decisión B de ADR-0015**
(que asumía RGB) · amplía ADR-0014 (taxonomía)

## Contexto

Con el backbone cerrado (audio F1 0,93 · asignación de equipo 86,83% = paper),
falta la contribución propia: clasificar el TIPO de golpe. ADR-0015 decidió
hacerlo por RGB, pero esa decisión se tomó **antes** de leer los papers enteros.
Leídos E2E-Spot (ECCV 2022), BST (CVPRW 2026) y Santra et al. (2025) —
bitácora en `docs/experiments.md` — el hallazgo obliga a corregir el rumbo:

**BST, el estado del arte en clasificación de tipo de golpe en deportes de
raqueta, NO usa RGB.** Usa pose (joints+bones) + trayectoria de pelota +
posición en pista, y bate a los modelos de esqueleto puro. La distinción que
faltaba: RGB gana en *detectar el instante* (PES, diferencias visuales sutiles
entre frames casi idénticos); pose+pelota gana en *clasificar el gesto*. Nuestro
instante ya lo da el AUDIO, más barato. Luego no necesitamos RGB.

Presupuesto de datos (techo, decisión de Jorge de no etiquetar más): 937
(PadelTracker100) + 413 (citys_cup) + 2.377 (CVSPORTS, instante ya marcado por
su GT, falta el tipo) = **~3.727 golpes**. Referencia: TenniSet tiene 3.569
golpes y 6 clases → 98,9% de accuracy. Estamos en ese régimen.

## Decisiones

- **A — Arquitectura: replicar BST** (como se hizo con el audio). Entradas:
  pose 2D + trayectoria de pelota + posición en pista, ya disponibles todas en
  nuestro backbone. Empezar por **BST-0** (pose+pelota, sin módulos CG/AP):
  en su tabla saca 0,8194 vs 0,8254 del completo — 99% del resultado con la
  mitad de complejidad. Baseline comparativo para la memoria: ST-GCN (Nivel 2),
  que en TenniSet (pocas clases, datos limpios) llega a batir a TemPose.
- **B — 4 clases: Saque, Derecha, Revés, Remate.** Se mantiene ADR-0014.
  Medido por BST sobre el MISMO dataset: 35 clases → acc 0,7626 / macro-F1
  0,6908; 25 clases fusionadas → **0,8284 / 0,8041**. +6,6 y +11 puntos solo por
  fusionar. Su criterio: fusionar lo que el modelo confunde (`wrist smash →
  smash`, `defensive return drive → drive`).
  **Regla adoptada: separar clases solo cuando difiere el GESTO, no la intención
  táctica ni la posición.**
  - **Volea NO es clase** (pregunta explícita de Jorge, volea-en-red vs volea
    atrás): una volea de derecha es un gesto de derecha, más corto y bloqueado —
    es el caso `defensive return drive → drive` que ellos fusionaron. Y no se
    pierde el dato: "volea" se deriva de `derecha + jugador cerca de la red`,
    usando la homografía (0,113 m). Atributo, no clase.
  - **Víbora NO es clase**: remate cortado, mismo gesto por encima de la cabeza
    (= `wrist smash → smash`).
  - **Bandeja NO es clase** (decisión de Jorge): el gesto sí difiere, pero
    saldrían pocos ejemplos y hundiría el macro-F1 como el `Other` a 0,18 del
    Nivel 2. Es más fácil añadir una clase después que rescatar un macro-F1
    hundido.
- **C — Atributos derivados, sin coste de etiquetado:** `de_pared` (ya se anota,
  ADR-0014 D) y `zona` (red/fondo, sale de la homografía). Dan granularidad
  analítica ("40% de los reveses de J3 vienen de pared", "voleas de J1") sin
  añadir clases al problema de clasificación.
- **D — Descartes: se etiquetan como `Other`, NO se entrenan, se usan para
  calibrar el rechazo.** Bandejas, víboras, dejadas y cualquier golpe atípico se
  marcan con una tecla de descarte y quedan en el CSV. El entrenamiento los
  filtra (un cajón de sastre visualmente incoherente envenena el modelo: no se
  parecen entre sí, y "brazo alto raro = Other" robaría remates). Ya entrenado,
  se le pasan los `Other` para ver qué confianza les da y fijar ahí el umbral.
- **E — En producción, "Sin clasificar" por umbral de confianza.** Decisión de
  Jorge: "prefiero que diga nada a que intente adivinar". Un softmax de 4 clases
  nunca dice "no sé" por sí solo, así que el rechazo se hace por umbral
  calibrado con los `Other` (Decisión D). Se guarda SIEMPRE la confianza en
  CSV/API aunque no se muestre (métrica para la memoria). Beneficio secundario:
  es red de seguridad de toda la cadena — si el audio detecta la pala contra el
  suelo, el clasificador verá una pose sin gesto de golpeo, dará confianza baja
  y saldrá "Sin clasificar" en vez de inventarse una derecha.
- **F — Ventana de recorte: adaptativa, no fija** (estrategia de BST §3.1).
  Del golpe anterior del rival al siguiente golpe del rival, + ε frames extra
  (ε = t/2, t = medio segundo) para capturar el inicio de la respuesta, lo que
  permite inferir el tipo hacia atrás. Medido por ellos: Min-F1 0,5210 → 0,5822
  (+6 puntos en la clase más difícil) frente a ventana de ancho fijo. Límite
  práctico suyo: no más de 1,5 s de distancia al golpe objetivo.
- **G — Etiquetado sobre el GT de CVSPORTS.** Sus 2.377 golpes traen el instante
  anotado (`hits.csv`), así que el anotador SALTA de golpe en golpe y Jorge solo
  pulsa el tipo. Desaparece la parte lenta (buscar el frame) y la fuente de
  ruido de ADR-0014 (puntería temporal del anotador).

## Consecuencias

- (+) No hay que montar pipeline de RGB (pesado, caro, muchos datos): las tres
  entradas que BST necesita ya existen en nuestro backbone.
- (+) 4 clases sobre ~3.700 golpes es mejor régimen que TenniSet (6 clases /
  3.569 → 98,9%). El objetivo de Jorge (>90% en golpes típicos) es alcanzable.
- (+) El rechazo por umbral es defendible ante el tribunal: el sistema conoce
  sus límites y están medidos, en vez de adivinar.
- (+) Los descartes dejan de ser basura: son el conjunto de calibración.
- (−) Se renuncia a distinguir bandeja/víbora/dejada. Aceptado a propósito
  (Jorge): la granularidad se puede añadir después si los datos acompañan.
- (−) Se renuncia al RGB como verificador de las dejadas que el audio no oye
  (decisión de Jorge: recall 0,87 con casi cero falsos positivos es suficiente;
  el coste/beneficio de un pipeline RGB para la clase más rara no compensa).
- (−) El calibrado del umbral necesita ~50-100 `Other`; si salen muy pocos, se
  fija mirando solo la distribución de los golpes buenos, con menos garantías.
- (−) Depende de la pose de nuestro YOLO y de la pelota (TrackNet F1 0,94): un
  golpe con pose rota entra ruidoso (mitigado por la tecla de descarte).

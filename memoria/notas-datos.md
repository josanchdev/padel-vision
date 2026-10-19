# Notas para el capítulo 3 (Datos)

Referencia de trabajo mientras se escribe el capítulo; se borra al terminar la memoria.
No es texto de la memoria: es la estructura propuesta, las cifras con su origen y los candidatos a
tabla o figura.

## Estado

- **18 oct:** PROPUESTA de estructura; falta que Jorge la valide.
- **`pages/datos.tex`:** vacío.

## Criterio (Jorge, 18 oct)

- **El capítulo trata de SU aportación:** el conjunto propio y su etiquetado.
- **CVSPORTS (Decorte et al.)** se menciona porque es la base sobre la que se etiqueta: hay que decir
  que se parte de ahí y qué aporta cada uno.
- **PadelTracker100 no va aquí:** ya se cita donde se usa, en la pelota (4.2.3) y en su evaluación
  (5.1.2).
- **El único etiquetador va aquí, como limitación,** en el párrafo del etiquetado (decidido al cerrar
  el capítulo 6).

## Estructura propuesta

**Introducción** (1-2 frases): de qué datos se parte y qué se añade en este trabajo.

### 3.1: qué decir (19 oct, un párrafo)

1. **De dónde sale y cómo se llama:** el conjunto publicado por Decorte et al.
   - ¡Ojo! «CVSPORTS» es el nombre que le damos en la memoria, por el taller en el que se presentó
     (CVsports, CVPRW 2024). En su repositorio se llama «Padel Hit Detection Dataset» (anotado en
     `notas_citas.md`). Hay que decirlo una vez aquí, para que el tribunal no pregunte qué es
     CVSPORTS.
2. **Qué contiene:** 99 puntos de los resúmenes de las retransmisiones de 11 torneos, grabados con la
   cámara fija de retransmisión y con audio.
   - Formato: todos a 1920×1080 y 25 fps (comprobado en los 99 vídeos, 19 oct); en total, 55,5 min de
     puntos.
   - El artículo habla de 5 h 28 min de resúmenes, de donde sacaron los puntos. No confundir las dos
     cifras.
3. **Qué anotaron ellos:**
   - el inicio y el fin del sonido de cada uno de los 2377 golpes;
   - el jugador que golpea, en los 319 golpes de 16 puntos de VIGO.

   Así, toda anotación ajena queda atribuida aquí, que es lo que pedía el padre de Jorge.
4. **La tabla de torneos** (`tab:dataset`): puntos, golpes y entorno.
5. **Lo que le falta,** que es el puente a la 3.2: el tipo de golpe.

### 3.1 Conjunto de partida: CVSPORTS

**Lo que aportan Decorte et al.** (citar una vez; dejar claro que lo anotaron ellos):
- 99 puntos de 11 torneos del World Padel Tour de 2023, grabados con la cámara de retransmisión y con
  audio;
- el instante de cada golpe (inicio y fin del sonido), 2377 golpes;
- el jugador que golpea, solo en 16 puntos de VIGO (319 golpes).

Lo que NO tiene, y por eso hace falta este trabajo: el tipo de golpe.

**Tabla candidata, que ya existe:** `figuras/tabla-dataset.tex` (`scripts/build_dataset_table.py`).
- Columnas: torneo, puntos, golpes y entorno (interior o exterior).
- Ahora no la usa ningún capítulo.
- La columna del entorno: VIGO, Bruselas y Valladolid son al aire libre. Esto respalda la frase de las
  conclusiones de que el clasificador no depende del aspecto de la pista: se evaluó dejando fuera
  torneos de interior y de exterior.

### 3.2 Etiquetado del tipo de golpe (la aportación)

- **Qué se etiqueta:** el tipo de cada uno de los 2377 golpes de CVSPORTS, sobre sus mismos instantes.
- **Criterio:** se agrupa por gesto.
  - Cuatro clases: derecha, revés, remate y saque.
  - Las voleas van como derecha o revés; la bandeja y la víbora, como remate.
  - «Other» solo para lo excepcional, que no entra en ninguna clase.
  - El porqué de las cuatro clases ya está en la 4.2.5; aquí solo el criterio.
- **La herramienta, propia** (`padel-cv annotate-types`): salta a cada instante anotado, reproduce el
  tramo alrededor y se elige la clase con una tecla; tiene barra de progreso.
  - **Figura candidata:** la interfaz, generada por código con el propio dibujo de la herramienta
    (`shot_type_annotator._draw_hud`) sobre un fotograma real.
  - El detalle de uso puede ir al anexo «Herramientas de etiquetado».
- **Limitación:** un único etiquetador, sin medida del acuerdo entre anotadores.
- **Resultado** (`dataset_shot_types.json`, todos los etiquetados):

  | Clase | Golpes |
  |---|---|
  | Derecha | 813 |
  | Revés | 735 |
  | Remate | 565 |
  | Saque | 97 |
  | Other | 167 (7 %) |
  | Total | 2377 |

  - **Desequilibrio de 8,4 a 1:** el saque es la clase menor. No es un sesgo del etiquetado: hay un
    saque por punto.
  - **Mediana de golpes por punto:** 23.
  - **Ojo:** los 2186 de la 5.1.4 son los que tienen jugador asignado. Aquí van los 2377 etiquetados.
- **Candidatos:** tabla o figura del reparto por clase (¿junto a la de torneos?), y una figura con un
  ejemplo de cada clase (fotograma del golpe con el jugador marcado, por código).

## Decisiones (18 oct)

- **Dos secciones.**
  - **3.1, «de dónde se parte»:** todo lo que es de Decorte et al., atribuido una vez y bien, para que
    lo de su padre quede resuelto en este capítulo.
  - **3.2:** todo lo propio (el criterio, la herramienta y el etiquetado).
- **Breve:** una página y media o dos, sobre todo tablas y figuras con poco texto.
- **El reparto por clase va como TABLA** (script nuevo): Clase | Golpes | Porcentaje, con una fila de
  total. Valores: 813 (34,2 %), 735 (30,9 %), 565 (23,8 %), 97 (4,1 %), 167 (7,0 %) y 2377.
  - «Otro» en vez de «Other» (en español, como las otras clases). Cambiarlo también en la 5.1.4 en la
    revisión final.
- **El reparto por torneo: FUERA** (a Jorge no le interesa).
- **La tabla de torneos en la 3.1: SÍ** (`tabla-dataset.tex`, `build_dataset_table.py`).
  - Es la tabla general del conjunto, como la tabla 1 de Decorte et al.: torneo, puntos, golpes y
    entorno.
  - Los recuentos salen de los ficheros del conjunto; el entorno, del artículo. Se cita en el texto,
    no en el título.
- **La figura de la herramienta: HECHA** (`build_annotator_figure.py` → `capturas/herramienta-etiquetado.jpg`).
  - Es la ventana real, dibujada con el propio código de la herramienta, sobre el remate de MALMO_02
    en el fotograma 615 (el mismo de la figura de la entrada).
  - Arriba a la izquierda, «Golpe 20 / 44», como quería Jorge.
  - La etiqueta de arriba a la derecha sale en inglés («Smash») y la leyenda pone «Reves» sin tilde.
    Se queda así (Jorge, 19 oct): «es una herramienta».
- **`pages/datos.tex`:** ya tiene las dos secciones, con las etiquetas `sec:conjunto-partida` y
  `sec:etiquetado`; todavía sin texto.
- **La tabla del reparto: HECHA** (`build_labels_table.py` → `tabla-etiquetado.tex`).
- **Corrección:** CVSPORTS NO lo grabaron Decorte et al. Son resúmenes de las retransmisiones de los
  torneos («each tournament's highlights», tabla 1 de su artículo). Ellos los recopilaron y anotaron
  los instantes y, en VIGO, el jugador.

## Por decidir con Jorge (lista inicial)

- ¿Dos secciones (partida y etiquetado) o texto seguido?
- ¿Qué tablas y figuras? La tabla de torneos (ya existe), el reparto por clase, la interfaz de la
  herramienta y los ejemplos por clase.
- ¿Qué va al capítulo y qué al anexo de herramientas?
- ¿Se menciona la otra herramienta propia, `annotate-court`, la del marcado de la pista? Lo más
  probable es que vaya al anexo.

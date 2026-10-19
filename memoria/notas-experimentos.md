# Notas para el capítulo 5 (Experimentos y validación)

Referencia de trabajo mientras se escribe el capítulo; se borra al terminar la memoria.
No es texto de la memoria: es la estructura acordada, lo que va en cada sección con sus cifras
exactas y su origen, las cautelas y los candidatos a tabla o figura que quedan por decidir.
Todo comprobado contra `docs/metrics/` (no contra `docs/sistema.md`, que sirve solo de base) y
contra el código de evaluación.

## Estado

**Estructura acordada (17 oct 2026, revisada el mismo día):**
- Introducción del capítulo, con los intervalos de confianza.
- 5.1 Resultados de cada componente.
- 5.2 El sistema completo.
- 5.3 Coste de procesado.

**Sin sección de protocolo, decisión de Jorge.** Cómo se mide cada componente va junto a su
resultado, porque cada uno usa sus propios datos y su propio criterio, y así el lector no tiene que
volver atrás. Solo lo común a todas las tablas, los intervalos de confianza, va en la introducción.
Recogido en `memoria/README.md`.

**Forma de trabajar:** sección a sección, como en el capítulo 4. Las tablas y figuras se deciden al
llegar a cada apartado; los candidatos de abajo son solo notas, no una propuesta cerrada.

**`pages/experimentos.tex`:** plantilla con las tres secciones y sus etiquetas
(`sec:resultados-componentes`, `sec:sistema-completo` y `sec:coste`), todavía sin texto.

| Sección | Tablas y figuras | Texto |
|---|---|---|
| Introducción del capítulo | — | **Hecho** (17 oct): contenido + intervalo, *bootstrap*, Monte Carlo y percentiles |
| 5.1.1 Instante del golpe | Sin tabla ni figura | **Hecho** (17 oct) |
| 5.1.2 Pelota | Sin tabla ni figura | **Hecho** (17 oct) |
| 5.1.3 Asignación del golpe | `tab:asignacion-paper` (hecha) | **Hecho** (18 oct) |
| 5.1.4 Tipo de golpe | `tab:tipo-clases` + `fig:tipo-confusion` (hechas: `build_shot_type_classes_table.py`, `build_confusion_figure.py`) | **Hecho** (18 oct). Ejes «Real»/«Predicho» (Jorge lo deja así) |
| 5.2 El sistema completo | `tab:cadena` (hecha) | **Hecho** (18 oct) |
| 5.3 Coste de procesado | `tab:equipo` + `tab:coste` (hechas: `build_hardware_table.py`, `build_cost_table.py`) | **Hecho** (18 oct); primero el equipo y después el coste |
| Pasada de conectores al estilo Miriam | — | **Hecha en todo el capítulo 5** (18 oct, 13 conectores) |

**CAPÍTULO 5 TERMINADO (18 oct 2026).**

## Reglas del capítulo

- **Cada tabla y figura sale de un script nuevo**, como en el capítulo 4.
  - El patrón es `scripts/build_*_table.py` → `memoria/figuras/tabla-*.tex`, leyendo el JSON de
    `docs/metrics/` y con los formatos de `padel_ml.thesis.latex`.
  - No se reutilizan las figuras de `docs/metrics/figures/`.
  - Tampoco el código anterior al reinicio de la memoria (`thesis/tables.py`, `thesis/figures.py`,
    `thesis/numbers.py` y `scripts/build_thesis_assets.py`). Ese código escribe en
    `figuras/generadas/`, que no usa ningún capítulo, y quizá se borre al final.
- **Las cifras de la prosa se comprueban contra el JSON.** Se usan los formatos de `latex.py`:
  - porcentajes con un decimal (87,5 %, no 87,46 %);
  - F1 y proporciones con tres decimales (0,958).
- **Aquí solo van los resultados del sistema final, cómo se miden y el coste.** La justificación de
  cada decisión ya está en el capítulo 4, junto a lo que justifica.
- **La comparación con Decorte et al. va solo en este capítulo** y solo donde se hace la misma prueba,
  que es la asignación.
  - Nada de «supera».
  - La detección «se reproduce» y es del mismo orden; no es una comparación exacta.
  - Nunca «más rápido que tiempo real».

## Las evidencias valen para el sistema actual (comprobado el 17 oct 2026)

**Cuándo se midieron.** Todo el 2 de octubre de 2026:

| Evidencias | Commit |
|---|---|
| Audio, asignación, tipo, cadena y pelota | `4dc5c7a` |
| Intervalos | `bc5bb35` |
| Coste | `baec97a` |

**Qué ha cambiado en el sistema desde entonces:**
- **Entre `4dc5c7a` y `baec97a`**, el refactor `1592fd3`. Se comprobó en GPU en su momento:
  - `analyze_rally` daba la misma huella en VIGO_11;
  - la tabla 2, re-puntuada, salía idéntica golpe a golpe.
- **De `baec97a` a hoy**, comparando el código sin docstrings ni comentarios (árbol sintáctico),
  solo cambian tres cosas, y ninguna interviene en lo medido:
  - `video_io.read_frame`, que se añadió;
  - una función nueva en `hit_assignment_gt`;
  - `rally_render`, el vídeo anotado, que no se mide.
- **`uv.lock` no ha cambiado desde `baec97a`**, así que el software es el mismo que el de la medida
  del coste.

**Conclusión:** no hace falta volver a medir nada.

## Estructura

### Introducción del capítulo

**Orden:**
1. Primero, como en los demás capítulos, qué se va a ver: los resultados de cada componente, el
   sistema completo y el coste de procesado.
2. Después, un párrafo corto sobre los intervalos de confianza, porque explica cómo leer todas las
   tablas que vienen.

**Los intervalos del 95 %**, lo único común a todas las tablas:
- **Qué son:** el rango en el que se movería la cifra si se midiera con otros puntos. Una cifra medida
  sobre pocos puntos (16 en la asignación) puede cambiar bastante con otros.
- **Cómo se calculan:** por *bootstrap* (`efron1979bootstrap`).
  - Se sacan al azar, con repetición, tantos puntos como hay en la prueba, y se vuelve a calcular la
    métrica.
  - Se repite muchas veces.
  - El intervalo va de los valores que dejan fuera un 2,5 % por cada lado.
- **Por qué se remuestrea el punto y no el golpe:** NO va en el texto (decisión de Jorge, 17 oct).
  Queda aquí solo por si lo pregunta el tribunal: los golpes de un mismo punto no son independientes
  (mismos jugadores, misma pista, mismo seguimiento), y remuestreando golpes el intervalo saldría más
  estrecho de lo que es. Sin la unidad en el texto, la excepción del tipo (remuestrea torneos)
  tampoco hace falta contarla.
- **Monte Carlo** (Jorge quiere mencionarlo, como Miriam):
  - Es correcto. Efron §2, «Method 2», aproxima la distribución del *bootstrap* por Monte Carlo con N
    réplicas, y está comprobado por Jorge en `notas_citas.md`.
  - La diferencia con Miriam: ella sortea los parámetros de entrada a partir de una distribución
    supuesta (normal, con las tolerancias) y los propaga por el modelo. Aquí se sortean los propios
    puntos de prueba, sin suponer ninguna distribución.
  - El porqué de Monte Carlo: la distribución del *bootstrap* no tiene fórmula cerrada para estas
    métricas, así que se aproxima con muchos remuestreos aleatorios.
  - Redacción acordada el 5 oct (`notas_citas.md`): «intervalos del 95 % por bootstrap [n], calculados
    por Monte Carlo con 10 000 remuestreos de rallies».
  - Monte Carlo se menciona una sola vez, dentro de esa frase.
  - En LaTeX, `\num{10000}`, que imprime 10 000 con espacio, como pide la RAE.
- **Qué no se atribuye a Efron:** el intervalo percentil (2,5 y 97,5), que se formalizó después. La
  cita va con «bootstrap».
- **Memoria o anexo:**
  - en la memoria: qué es, el 95 %, el *bootstrap* por puntos y por qué, Monte Carlo y los 10 000
    remuestreos, con la cita a Efron, en palabras y sin ecuación;
  - en el anexo de reproducibilidad: la semilla y el método de percentiles.
- **Extensión:** un párrafo en la introducción, sin sección propia (decidido el 17 oct). Donde de
  verdad pesa es en la 5.1: la comparación emparejada con el paper.
- **Las excepciones se cuentan donde ocurren**, en la 5.1: el tipo de golpe remuestrea torneos, y la
  detección no lleva intervalo.

`efron1979bootstrap` está en el `.bib` y ya revisada en `notas_citas.md` (Descargas/papers_tfg).

### 5.1 Resultados de cada componente

**Qué cuenta:** cada paso medido por separado, cada uno con cómo se mide; en la asignación, además,
junto al paper.

**Organización decidida (17 oct):**
- Subsecciones con el orden y los nombres de la 4.2: Instante del golpe, Pelota, Asignación del golpe
  y Tipo de golpe. El tipo es lo más importante y el que más espacio lleva.
- Ya están en el `.tex`, con las etiquetas `sec:resultados-instante`, `-pelota`, `-asignacion` y
  `-tipo`.
- Sin tabla global.
- Jugadores no tiene subsección: basta una frase en la asignación diciendo que no hay anotación propia
  y que se evalúa a través de ella.

**Hallazgos al preparar el detalle (17 oct):**
- **Intervalos que no existen.** RESUELTO: Jorge cambió la intro a «las cifras principales». La intro
  decía «cada cifra se acompaña de un intervalo», pero no tienen intervalo:
  - la detección (lleva la desviación de 3 ejecuciones);
  - la pelota;
  - el macro-F1;
  - las cifras por clase.

  `confidence_intervals.json` solo cubre la asignación, la *accuracy* del tipo y la cadena.
- **La limpieza de la trayectoria no se describe en el capítulo 4.** Las cifras «con limpieza» de la
  pelota necesitarían explicarse; mejor no usarlas.
- **Coherencia con el capítulo 4.** `tab:deteccion` da 0,941 para el audio (5 puntos de VIGO, para
  compararlo con la imagen) y aquí sale 0,958 (3 ejecuciones). Hace falta una frase que lo aclare.
- **Errores de la asignación** (`hit_assignment_replica.confusion`): 40 fallos de 319.
  - 24 van a un jugador de la otra pareja, 14 al compañero y 2 se quedan sin asignar.
  - Ojo: la mayoría NO son confusiones con el compañero.
  - La alternancia del equipo (como en el paper) recupera el equipo de 1 de los 2 golpes sin asignar.
    Es despreciable y no hace falta contarlo.
- **Asignación por punto, frente a la tabla 3 del paper** (`PAPER_TABLE_3` en `hit_assignment_gt`):
  - gana en 6 puntos: 01 (100 frente a 83,9), 03, 06, 09 (97,1 frente a 77,1), 11 (100 frente a 87,0)
    y 15;
  - empata en 5: 02, 05, 08, 12 y 16;
  - pierde en 5: 00 (43,8 frente a 87,5, por el cruce de identidades), 04, 07, 10 y 14;
  - no existe el VIGO_13.
- **Tipo, de dónde salen los 2186 golpes:** de los 2377 etiquetados se quitan los 167 «Other» y quedan
  2210 de las cuatro clases; de esos, 24 no tienen jugador asignado.
- **Tipo, errores:** 378 en total.
  - 238 (63 %) entre derecha y revés;
  - 132 (35 %) del remate con derecha o revés;
  - 8 saques.
- **Sin la regla del saque**, salen 101 saques predichos con 87 correctos: 14 falsos, que la regla
  elimina.
- **Golpes de las cuatro clases por torneo** (`dataset_per_tournament`):

  | Torneo | Golpes | Torneo | Golpes |
  |---|---|---|---|
  | VIGO | 294 | Madrid | 279 |
  | Bruselas | 227 | Alemania | 219 |
  | Valladolid | 107 | Ámsterdam | 171 |
  | Valencia | 98 | Menorca | 183 |
  | Málaga | 188 | Malmö | 156 |
  | Finlandia | 288 | | |

  El acierto no tiene una relación clara con el tamaño del torneo.

#### Detalle de cada subsección (explicación de trabajo, 17 oct)

Es la explicación completa de qué se mide, qué sale y qué se puede comentar en cada subsección, con
una sugerencia de tabla o figura. Las sugerencias no están decididas: se deciden con Jorge al llegar
a cada subsección. Debajo de este bloque siguen los datos de referencia (tablas con todas las cifras).

##### Maneras de empezar cada subsección (17 oct)

**Cómo se organiza cada subsección.**
- **Mismo orden de ideas en las cuatro:**
  1. cómo se mide;
  2. el resultado;
  3. la lectura;
  4. la comparación o la cautela, donde la haya.
- **Frases distintas en cada una.** El paralelismo lo da el orden, no repetir la misma plantilla.
- **Tiempo verbal:** presente, como en la introducción («se miden», «se utiliza»).
- **Evitar el anuncio vacío** («En esta subsección se presentan…»): el título ya lo dice.

**Aperturas posibles:**

| Arranque | Ejemplo |
|---|---|
| Por el objetivo | «Para evaluar la detección del instante del golpe se emplean los 99 puntos de CVSPORTS, que incluyen el instante anotado de cada golpe.» |
| Por los datos | «El conjunto CVSPORTS dispone del instante anotado de los 2377 golpes de sus 99 puntos, lo que permite medir directamente la detección.» |
| «La evaluación de… se realiza sobre…» | «La evaluación del detector de pelota se realiza sobre la final femenina de PadelTracker100, que el modelo no ha visto durante el entrenamiento.» |
| Por una condición del dato | «Dado que solo los 16 puntos de VIGO tienen anotado qué jugador golpea, la asignación se mide sobre sus 319 golpes.» |
| Por la causa | «Puesto que el detector de audio necesita entrenarse, los puntos se dividen al azar en dos grupos: uno para entrenar y otro para medir.» |
| Por contraste con lo anterior | «A diferencia de los componentes anteriores, la asignación no requiere entrenamiento, por lo que puede medirse sobre todos los golpes anotados.» |
| Por el criterio de acierto | «Un golpe se considera bien detectado cuando cae a menos de 250 ms de uno anotado.» |
| Por la pregunta que responde | «El objetivo de esta prueba es comprobar si el clasificador mantiene su acierto en pistas que no ha visto.» |
| Por el protocolo de referencia (solo detección y asignación) | «Siguiendo el protocolo de Decorte et al.~\cite{decorte2024padel}, la asignación se mide sobre los mismos 319 golpes de los 16 puntos de VIGO.» |
| Por las métricas | «El rendimiento del clasificador se mide mediante la \textit{accuracy} y el macro-F1, que da el mismo peso a las cuatro clases.» |
| Por la cantidad | «Un total de 2186 golpes etiquetados forman la base de la evaluación del tipo de golpe.» |
| Por cómo se obtienen los resultados | «Los resultados de la pelota se obtienen entrenando el detector con la final masculina de PadelTracker100 y evaluándolo con la femenina.» |

##### Instante del golpe

**Cómo se mide.**
- **Datos:** CVSPORTS, que trae anotado el instante de los 2377 golpes de sus 99 puntos.
- **Separación:** el detector de audio tiene que entrenarse, así que los puntos se reparten al azar,
  70 para entrenar y 29 para medir.
- **Tres ejecuciones:** una sola partición puede salir más fácil o más difícil por azar, así que el
  entrenamiento se repite tres veces, cada una con su propio reparto y sus propios pesos iniciales. La
  cifra final es la media de las tres, con su desviación típica.
- **Sin intervalo:** esa desviación hace aquí el papel del intervalo, y por eso esta cifra no lleva
  intervalo de *bootstrap*. Hay que decirlo en el texto, porque la introducción anuncia intervalos para
  las cifras principales.
- **Acierto:** un golpe detectado cuenta como acierto si cae a menos de 250 ms de uno anotado, y cada
  golpe anotado solo puede emparejarse con una detección. Se cuentan los aciertos, las falsas alarmas
  y los golpes que se escapan, y de ahí salen la precisión, el *recall* y el F1.

**Qué sale.** F1 de 0,958 ± 0,007, con precisión 0,972 y *recall* 0,945.
- La desviación es muy pequeña, así que el resultado no depende de qué puntos tocaron en la prueba.
- La precisión es mayor que el *recall*: cuando el detector marca un golpe, casi siempre lo hay; lo que
  falla es más bien que se le escape alguno.

**ESTADO (17 oct): texto escrito por Jorge.**

**DECIDIDO (Jorge, 17 oct): en la detección no se menciona el paper.** No se compara lo mismo, y
mencionarlo invita a preguntar por qué no se usó su criterio.
- **Si lo pregunta el tribunal:** el paper da 0,92 con la métrica por eventos (inicio y fin del sonido).
  El sistema entrega un instante por golpe, no un intervalo de sonido, así que se mide lo que entrega.
  Con criterios distintos solo se puede decir que es del mismo orden.
- La explicación de abajo queda solo como referencia.

**La comparación con el paper (descartada para el texto).** Decorte et al. obtienen 0,92 con los mismos
datos, pero hay dos diferencias de protocolo:
- **Cómo se cuenta un acierto.** El paper usa la métrica por eventos de sed_eval (Mesaros et al.,
  `mesaros2016metrics`), que compara el inicio y el fin de cada sonido. Aquí solo se compara el
  instante.
- **Cuántas particiones se promedian:** cuatro en el paper, tres aquí.

Con criterios distintos, 0,958 frente a 0,92 no demuestra una mejora. Lo que sí muestra es que la
reimplementación llega al mismo orden, es decir, que el método se reproduce.

**Coherencia con el capítulo 4.** La tabla de detección por imagen y por audio de la 4.2.1
(`tab:deteccion`) da 0,941 para el audio.
- **DECIDIDO (Jorge, 17 oct): no se menciona en la 5.1.** El título de esa tabla ya dice «en los
  mismos cinco puntos», y el lector no tiene que volver atrás.
- **Si lo pregunta el tribunal:**
  - el 0,941 es el detector de la ronda VIGO de la cadena (entrenado sin VIGO), medido en 5 puntos para
    comparar audio e imagen sobre los mismos 94 golpes;
  - la imagen no se puede medir en más puntos: su modelo se retiró y solo vive en `pre-limpieza`;
  - 0,958 es la media de 3 particiones con todos los puntos;
  - 0,948 (5.2) es la cifra con un torneo nuevo.

**Sugerencia: ni tabla ni figura.**
- Son tres cifras y una referencia, que caben en dos o tres frases; una tabla de una fila no aporta.
- La figura del barrido del umbral no va aquí: justificaría una decisión, y eso pertenece al
  capítulo 4, donde el umbral ya está como «fijado empíricamente».

##### Pelota

**Cómo se mide.**
- **Datos:** el detector se entrena con PadelTracker100, que anota la posición de la pelota en cada
  fotograma de las dos finales de Barcelona 2022.
- **Separación:** se entrena con la final masculina y se mide en la femenina, que no ha visto. Son
  45 934 fotogramas, 37 904 con la pelota visible; los demás sirven para ver si el detector se inventa
  la pelota donde no está.
- **Condiciones:** se mide tal como funciona dentro del sistema, a su resolución de entrada.
- **Acierto:** una detección es correcta si cae a menos de 15 px de la posición anotada, en la imagen de
  1080p. La precisión dice qué parte de las detecciones son correctas, y el *recall*, en qué parte de
  los fotogramas con pelota la encuentra.

**Qué sale.** F1 de 0,915, con precisión 0,871 y *recall* 0,964. El error mediano es de 8,1 px: la mitad
de las detecciones caen a menos de 8 px de la pelota real.
- El *recall* alto indica que encuentra la pelota en casi todos los fotogramas en los que está.
- La precisión menor indica que algunas detecciones son falsas.
- Al ampliar la tolerancia a 30 o 60 px, el F1 apenas sube (0,932 y 0,933). Cuando encuentra la pelota,
  la sitúa bien; lo que falla son detecciones en otro sitio o en fotogramas sin pelota.

**Lo que hay que advertir.**
- Las dos finales se jugaron en la misma pista y con la misma cámara, así que esto no mide cómo se
  comporta el detector en otras pistas.
- Esta cifra no tiene intervalo.

**Lo que se deja fuera.**
- **Las cifras «con limpieza de trayectoria»:** el capítulo 4 no describe esa limpieza, así que habría
  que explicarla antes. Sin ellas se entiende igual.
- **El paper:** no aparece, porque usa un modelo de tenis sin medirlo en pádel y no hay nada que
  comparar.

**Sugerencia: solo prosa.**
- El mensaje es una cifra (F1 de 0,915 a 15 px) y una observación (ampliar la tolerancia casi no cambia
  nada), y cabe en un párrafo.
- Alternativa: una tabla pequeña con 15, 30 y 60 px en filas, y precisión, *recall* y F1 en columnas.
  Pero de sus nueve números solo importan tres.

##### Asignación del golpe

**Cómo se mide.**
- **Datos:** quién golpea solo está anotado en 16 puntos de VIGO, 319 golpes. Son exactamente los de la
  tabla 3 del paper, con el mismo número de golpes en cada punto.
- **Instantes:** se usan los **anotados**, no los del audio, para medir este paso sin los errores de la
  detección. La combinación de los dos se mide en la 5.2.
- **Separación:** el voto ponderado no se entrena, así que no hace falta apartar datos de prueba.
- **Acierto:** la proporción de golpes atribuidos al jugador correcto. Los golpes sin asignar cuentan
  como fallo, igual que en el paper. En la pareja basta con que el jugador elegido sea del mismo
  equipo.
- **Frase de los jugadores (va aquí):** su detección y su seguimiento no tienen anotación propia; su
  calidad se refleja en esta asignación.

**Qué sale.** El jugador acierta en el 87,5 % de los golpes (78,4-93,6) y la pareja en el 92,2 %
(84,4-96,8).
- Los intervalos son anchos, unos quince puntos, porque solo hay 16 puntos.
- De los 40 fallos, 24 van a un jugador de la otra pareja, 14 al compañero y 2 golpes se quedan sin
  asignar. La mayoría de los errores no son confusiones con el compañero.

**La comparación con el paper (la única directa).**
- **Las medias:** Decorte et al. obtienen 83,7 % en jugador y 86,8 % en pareja, así que este trabajo
  queda 3,8 y 5,3 puntos por encima.
- **La pregunta que importa:** no es si la media es mayor, sino si la diferencia es real. Para
  responderla se usa el mismo *bootstrap* de la introducción, de forma **emparejada**: en cada
  remuestreo se eligen los mismos puntos para los dos métodos, y así la dificultad propia de cada punto
  se cancela.
- **El resultado:** el intervalo de la diferencia va de −5,4 a +9,9 puntos en jugador y de −2,9 a +10,2
  en pareja. Los dos incluyen el cero, así que no se puede afirmar que sea mejor. Sí se puede afirmar
  que el método se reproduce con un acierto al menos comparable.
- **Punto a punto se ve por qué:**
  - gana en 6 puntos, sobre todo en los largos (VIGO_09: 97,1 % frente a 77,1 %; VIGO_11: 100 % frente
    a 87,0 %);
  - empata en 5;
  - pierde en 5, con el caso más claro en VIGO_00 (43,8 % frente a 87,5 %), por un cruce de identidades
    entre dos jugadores.

  La ventaja no es constante, y eso es lo que hace que el intervalo incluya el cero.

**DECIDIDO (Jorge, 18 oct, final): la cautela del optimismo NO va en el texto.** Si lo pregunta el
tribunal: las mejoras se eligieron sobre estos 319 golpes, porque no hay otros con jugador anotado.

**La cautela (descartada para el texto).**
- Las mejoras de la asignación se eligieron midiendo sobre estos mismos 319 golpes: la distancia
  dividida por la altura del jugador, la ventana de ±4 fotogramas y la resolución de la pelota.
- Sus justificaciones están en la 4.2.4, pero la cifra puede ser algo optimista.
- Decirlo es lo que hace creíble la comparación.

**TABLA HECHA (17 oct), con el diseño de Jorge.**
- **Columnas:** Acierto | Este trabajo | Intervalo 95 % | Decorte et al. | Diferencia. La diferencia va sin
  su intervalo: Jorge no quiere paréntesis.
- **Ficheros:** `scripts/build_assignment_comparison_table.py` → `figuras/tabla-asignacion-paper.tex`,
  etiqueta `tab:asignacion-paper`. El título es provisional.
- **Porcentajes con un decimal**, la convención de la memoria.
- **El intervalo de la diferencia** (de −5,4 a +9,9 y de −2,9 a +10,2), que lleva la conclusión, va
  en la prosa.
- **Figura por punto:** DESCARTADA (Jorge, 17 oct).

**Sugerencia original: una tabla de comparación.**
- Dos filas (jugador y pareja) y tres columnas: este trabajo con su intervalo, Decorte et al., y la
  diferencia con su intervalo.
- Es la tabla más importante de la subsección: enseña a la vez la mejora en la media y que no es
  concluyente.
- Opcional: una figura con el acierto en cada uno de los 16 puntos, el propio frente al del paper. Solo
  merece la pena si se quiere contar lo de VIGO_00 y lo de los puntos largos; si no, sobra.

##### Tipo de golpe

**Cómo se mide.**
- **Datos:** el etiquetado propio, 2377 golpes.
  - Se quitan los 167 marcados como «Other», que son los casos excepcionales.
  - De los 2210 restantes se quitan 24 sin jugador asignado: sin jugador no hay esqueleto que
    clasificar.
  - Quedan 2186.
- **Separación: un torneo fuera cada vez.** En cada una de las 11 rondas, el clasificador se entrena con
  diez torneos y se mide en el que queda. Al final, cada golpe ha sido clasificado una vez, por un
  modelo que nunca vio su torneo.
- **Por qué no se reparten los puntos al azar:** los puntos de un mismo torneo comparten pista, cámara e
  iluminación. Al azar, el modelo vería en el entrenamiento la misma pista con la que luego se le mide,
  y la cifra saldría optimista. Así responde a lo que importa: si funciona en una pista que nunca ha
  visto.
  - Es la primera vez que aparece este protocolo, así que se explica aquí; en la 5.2 basta con
    recordarlo.
- **Métricas:**
  - la *accuracy*;
  - el macro-F1, la media del F1 de las cuatro clases. Sin él, el saque, con solo 95 golpes, apenas
    pesaría; con el macro-F1 pesa lo mismo que las demás clases.

**Qué sale.** *Accuracy* del 82,7 % (80,3-85,0) y macro-F1 de 0,859.
- **Por clase:** el saque es el mejor (F1 0,956), le sigue el remate (0,879), y la derecha y el revés se
  quedan casi iguales (0,805 y 0,796).
- **La matriz de confusión explica la diferencia.** De los 378 errores:
  - 238 (el 63 %) son entre derecha y revés;
  - 132 (el 35 %) son del remate con derecha o revés;
  - solo 8 son saques.

  El problema es separar derecha de revés; el remate se distingue bastante bien y el saque casi siempre.
- **La causa, solo como hipótesis** (no está medida): es el mismo gesto en el lado contrario del cuerpo,
  y en un jugador zurdo la derecha cae en el lado del revés de un diestro. Recomendación: describir lo
  que se ve, sin atribuirle una causa.

**Por torneo.** El acierto va del 75,4 % (Ámsterdam) al 91,7 % (Malmö).
- No hay relación clara con el tamaño del torneo:
  - Valencia, con 98 golpes, se queda en el 76,3 %;
  - Malmö, con 156, llega al 91,7 %;
  - Ámsterdam, con 171, se queda en el 75,4 %.
- Muestra que la pista y la cámara influyen, y esa variación es la que recoge el intervalo.

**La regla del saque, medida.** La regla se explica en la 4.2.5; aquí va su efecto.
- **Sin la regla:** el clasificador predice 101 saques, de los que solo 87 lo son, es decir, una
  precisión de 0,861.
- **Con la regla:** desaparecen los 14 falsos, porque ninguno era el primer golpe del punto, y la
  precisión llega a 1,000.
- **El *recall* no cambia (0,916):** los 8 saques que se escapan sí eran el primer golpe, pero el modelo
  los tomó por otra clase.
- **En total:** el macro-F1 sube de 0,842 a 0,859 y la *accuracy* de 82,5 % a 82,7 %.

**Los pesos de clase** (pendiente desde el capítulo 4; va aquí, con `\cref{sec:tipo}`).
- Dar más peso en el entrenamiento a las clases con menos ejemplos es lo habitual cuando las clases
  están desequilibradas.
- Se probó con dos intensidades: el macro-F1 quedó en 0,855 y 0,857, frente a 0,859 sin pesos.
- Con la regla del saque no aportan nada, así que el clasificador se entrena sin ellos.

**DECIDIDO (Jorge, 18 oct).** Mañana termina de redactar; los scripts se hacen cuando lo pida.
1. **Tabla por clase:** sí. Columnas Clase | Precisión | *Recall* | F1 | Golpes, con una fila final
   «Media (macro)» (solo F1 0,859 y 2186 golpes).
2. **Matriz de confusión como figura, hecha de nuevo con un script.**
   - **Estilo:** la matriz «de toda la vida», la que todo el mundo reconoce (sklearn / MLflow).
     NO sigue el estilo de las figuras del capítulo 4.
   - **Detalles:**
     - mapa de color azul (Blues) con su barra de color;
     - filas = real y columnas = predicho;
     - en cada casilla, la proporción de la fila con 3 decimales y debajo el número de golpes entre
       paréntesis.
   - **Referencia:** `docs/metrics/figures/shot_type_confusion.png` (Jorge la pasó como ejemplo), que
     NO se reutiliza.
   - **Confirmado por Jorge (18 oct):**
     - coma decimal (0,803) para ir con el texto;
     - sin título dentro de la imagen: lo da el `\caption`;
     - etiquetas de los ejes «Real» y «Predicho».
3. **Por torneo:** solo el rango en la prosa.
4. **Regla del saque y pesos de clase:** solo prosa.

- **Columna «sin clasificar»:** no. En esta evaluación no existe: los 2186 golpes reciben una de las
  cuatro clases (la matriz suma 2186). Los 24 sin jugador se quitan antes, y eso se dice en el texto.

**Sugerencias:**
- **Tabla por clase**, con precisión, *recall*, F1 y número de golpes, y una fila final para la media
  macro: sí. Da las cifras exactas de la aportación.
- **Matriz de confusión como figura:** sí. Es lo que mejor enseña que el error se concentra en
  derecha ↔ revés, y se ve de un vistazo. No repite la tabla: la tabla da las cifras y la figura enseña
  dónde van los errores.
- **Por torneo:** el rango en la prosa basta. Una figura de once barras, solo si se quisiera comentar
  torneos concretos.
- **Regla del saque y pesos de clase:** en la prosa. Son cuatro cifras y dos cifras, y una tabla no
  aporta.

#### Datos de referencia de la 5.1

**Cómo se mide cada uno** (va en el texto, junto a su resultado):

- **Detección del golpe.**
  - **Datos:** CVSPORTS (Decorte et al.), 99 puntos de 11 torneos con 2377 golpes de instante anotado.
  - **Separación:** 3 ejecuciones (semillas 0, 1 y 2), cada una con su propio reparto aleatorio de los
    puntos (70 de entrenamiento y 29 de prueba) y sus propios pesos iniciales. Se da la media ± la
    desviación típica, que equivale a promediar particiones, como hace el paper con 4.
  - **Intervalo:** no lleva. La variación ya la da la desviación típica de las 3 ejecuciones.
  - **Acierto:** un golpe detectado es correcto si está a menos de 250 ms de uno anotado, con
    emparejamiento uno a uno (`audio_train.match_events`). Se dan precisión, *recall* y F1.
  - **El paper** usa la métrica por eventos de sed_eval, que mira el inicio y el fin del evento, con
    250 ms o un 50 % de solape (`mesaros2016metrics`, en el `.bib` y en `notas_citas.md`).
- **Asignación del golpe al jugador.**
  - **Datos:** los 16 puntos de VIGO (319 golpes). Es el único torneo en el que el paper anotó quién
    golpea.
  - **Separación:** no hay entrenamiento, porque es el voto ponderado.
  - **Instantes:** se mide con los instantes **anotados**, no con los detectados, para aislar este
    paso.
  - **Acierto:** se pondera por golpes, es decir, es la proporción de golpes atribuidos al jugador
    correcto. Los golpes sin asignar cuentan como fallo, en los dos trabajos.
  - **Pareja:** J1-J2 frente a J3-J4 (`chain_eval.same_team`).
- **Tipo de golpe.**
  - **Datos:** el etiquetado propio, que se describe en el capítulo Datos. Entran 2186 golpes: los de
    las cuatro clases que tienen jugador asignado.
  - **Separación:** un torneo fuera cada vez, en 11 rondas. Los puntos de un torneo comparten pista,
    cámara e iluminación, así que repartir por puntos filtraría. La cifra responde a si el
    clasificador funciona en una pista que nunca ha visto.
  - **Ojo:** como esta es la primera vez que sale «un torneo fuera cada vez», el porqué se explica
    aquí; en la 5.2 basta con recordarlo.
  - **Intervalo:** se remuestrean torneos (11), no puntos, porque la validación cruzada guarda sus
    resultados por torneo. Sale más ancho, es decir, más prudente.
  - **Acierto:** *accuracy* y macro-F1. El macro-F1 es la media del F1 de las cuatro clases, para que el
    saque (95 golpes) pese lo mismo que las demás.
- **Pelota.**
  - **Datos:** PadelTracker100. El detector se entrena con la final masculina y se mide en la femenina:
    45 934 fotogramas, 37 904 de ellos con pelota.
  - **Acierto:** F1, contando como correcta una detección a menos de 15 px de la anotada, en 1080p.

Los hiperparámetros de entrenamiento van al anexo de reproducibilidad, como se decidió en la 4.2.5.
Por ejemplo, para el clasificador: 60 épocas, lotes de 64 y una tasa de aprendizaje de 0,0005.

**Cifras** (base: la tabla de componentes de `sistema.md`):

| Medida | Este trabajo | Decorte et al. | JSON |
|---|---|---|---|
| Detección (F1) | 0,958 ± 0,007 | 0,92 | `audio_threshold_seeds` (`f1`, `f1_std`, `paper_f1`) |
| Asignación: jugador | 87,5 % (78,4-93,6) | 83,7 % | `confidence_intervals` + `hit_assignment_replica` |
| Asignación: pareja | 92,2 % (84,4-96,8) | 86,8 % | ídem |
| Tipo: *accuracy* | 82,7 % (80,3-85,0) | no lo hace | `confidence_intervals` + `shot_type_classifier` |
| Tipo: macro-F1 | 0,859 | no lo hace | `shot_type_classifier` |
| Pelota: F1 a 15 px | 0,915 | no lo mide en pádel | `ball_detector` |

- **Detección, desglose** de la media de 3 ejecuciones:
  - precisión 0,972 y *recall* 0,945;
  - el umbral es 0,5, el mejor por F1 medio.
- **Pelota, desglose:**
  - precisión 0,871, *recall* 0,964 y error mediano de 8,1 px;
  - F1 de 0,932 a 30 px y de 0,933 a 60 px;
  - con la limpieza de trayectoria (la que usa el voto): F1 de 0,902 a 15 px y *recall* de 0,975;
  - el 0,942 de la validación de entrenamiento (512×288, unos 60 px) **no** se cita.

**Cómo leer la comparación con el paper** (las notas de `sistema.md`, para pasar a prosa):

1. **Asignación: comparación directa, resultado comparable.**
   - Son los mismos 319 golpes (los 16 puntos de la tabla 3 del paper, con los mismos golpes en cada
     uno) y la misma métrica.
   - Quedamos +3,8 puntos por encima en jugador y +5,3 en pareja.
   - Pero el intervalo de la diferencia emparejada incluye el cero: de −5,4 a +9,9 en jugador y de
     −2,9 a +10,2 en pareja. La diferencia es emparejada porque se remuestrean a la vez los mismos
     16 puntos para los dos métodos, y así la dificultad de cada punto se cancela.
   - Con 16 puntos, la mejora **no es concluyente**. Lo que sí se sostiene es que el método se
     reproduce con un acierto al menos comparable.
2. **Las mejoras de la asignación se eligieron midiendo sobre esos mismos 319 golpes**: la distancia
   dividida por la altura, la ventana de ±4 fotogramas y la resolución de la pelota. Por eso la cifra
   puede ser algo optimista.
3. **Detección: mismo orden, no una comparación exacta.** (DESCARTADO para el texto: en la detección
   no se menciona el paper.)
   - El paper empareja eventos por inicio y fin y promedia 4 particiones.
   - Aquí se empareja por instante y se promedian 3.
   - 0,958 frente a 0,92 no demuestra una mejora.
4. **El tipo y la pelota no se comparan**: el paper no clasifica golpes ni mide su detector de pelota
   en pádel.

**Detalle del clasificador** (la aportación; propuesto, falta decidir qué entra):

- **Por clase**, con la regla del saque (`shot_type_classifier.per_class`):

  | Clase | Precisión | *Recall* | F1 | Golpes |
  |---|---|---|---|---|
  | Derecha | 0,806 | 0,804 | 0,805 | 799 |
  | Revés | 0,789 | 0,803 | 0,796 | 731 |
  | Remate | 0,882 | 0,877 | 0,879 | 561 |
  | Saque | 1,000 | 0,916 | 0,956 | 95 |
  | Media (macro) | — | — | 0,859 | 2186 |

- **Matriz de confusión** (`shot_type_classifier.confusion`). Las filas son la clase real y las
  columnas la predicha, en el orden derecha, revés, remate, saque:

  |  | Derecha | Revés | Remate | Saque |
  |---|---|---|---|---|
  | **Derecha** | 642 | 123 | 34 | 0 |
  | **Revés** | 115 | 587 | 29 | 0 |
  | **Remate** | 39 | 30 | 492 | 0 |
  | **Saque** | 1 | 4 | 3 | 87 |

  Lo que se ve:
  - la confusión principal es derecha ↔ revés (123 y 115);
  - el remate se confunde poco;
  - gracias a la regla, ningún golpe se toma por saque sin serlo;
  - se pierden 8 saques.

- **Por torneo** (`shot_type_per_tournament`): de 75,4 % (Ámsterdam) a 91,7 % (Malmö).

  | Torneo | *Accuracy* |
  |---|---|
  | VIGO | 84,7 % |
  | Bruselas | 78,5 % |
  | Valladolid | 81,1 % |
  | Valencia | 76,3 % |
  | Málaga | 86,2 % |
  | Finlandia | 83,4 % |
  | Madrid | 80,5 % |
  | Alemania | 85,8 % |
  | Ámsterdam | 75,4 % |
  | Menorca | 83,1 % |
  | Malmö | 91,7 % |

- **Regla del saque** (ya explicada en la 4.2.5, `eq:regla-saque`; aquí va su efecto medido):

  | | *Accuracy* | Macro-F1 | Precisión del saque | *Recall* del saque |
  |---|---|---|---|---|
  | Sin la regla | 82,5 % | 0,842 | 0,861 | 0,916 |
  | Con la regla | 82,7 % | 0,859 | 1,000 | 0,916 |

- **Pesos de clase** (pendiente desde el capítulo 4; va donde se cuente el entrenamiento, con
  `\cref{sec:tipo}`):
  - macro-F1 de 0,859 sin pesos, 0,855 con potencia 0,5 y 0,857 con potencia 1;
  - con la regla del saque dan lo mismo, así que se entrena sin pesos.

**Candidatos por decidir:**
- Tabla de componentes con la columna del paper: **es la base**, sí entra.
- La pelota: como fila de esa tabla (con «—» en la columna del paper) o en una tabla aparte, como en
  `sistema.md`.
- Tabla por clase.
- Matriz de confusión: como figura o como tabla.
- Acierto por torneo: tabla de 11 filas, figura de barras o solo el rango en la prosa.
- Regla del saque: una tabla pequeña con y sin ella, o en la prosa.

### 5.2 El sistema completo

**Qué cuenta:** la cadena entera tal como funciona en uso. El audio decide cuándo; el voto decide
quién, en el instante que encontró el audio; y el clasificador decide qué tipo.

#### Estructura del texto (decidida por Jorge, 18 oct)

**Orden:** introducción (ya escrita; sus comentarios `%` los arregla Jorge al final) → cómo se mide →
`tab:cadena` → conclusiones breves. Nada más: ni figura ni embudo aparte, porque el embudo de VIGO
ES el bloque VIGO de la tabla.

**Cómo se mide:**
- un torneo fuera cada vez, como en la 5.1.4, y ningún modelo vio el torneo;
- cada etapa cuenta solo los golpes que superan también las anteriores, sobre los golpes reales;
- dos bloques: los 11 torneos (cuándo y qué) y VIGO (la cadena entera, con el quién).

**Conclusiones (2-4 frases):**
- lo que se pierde en cada etapa en VIGO: de los 286 golpes, 11 al detectar, 33 al asignar y 26 al
  clasificar.
  - ¡Ojo! En puntos sale 3,8, 11,5 y 9,1. Restando los porcentajes redondeados de la tabla sale 11,6,
    así que es mejor dar los golpes o no dar cifras.
- las falsas alarmas: el 2,8 % de las detecciones no son golpes reales (1 − 0,972);
- opcional: el tipo acierta el 89,3 % cuando el jugador es correcto, frente al 85,1 % en VIGO;
- el resumen: tres de cada cuatro golpes de VIGO salen del todo correctos.

#### TABLA HECHA (18 oct): `tab:cadena`

- **Ficheros:** `scripts/build_chain_table.py` → `figuras/tabla-cadena.tex`.
- **Cifras:** las intermedias se añadieron a `scripts/evidence_confidence_intervals.py` (funciones
  `stages` y `chain_hits`) y el JSON se regeneró. Ninguna cifra anterior cambió.
- **El JSON está regenerado desde un commit limpio, `c6f7420`.** Commits del 18 oct: `c6f7420`
  (scripts), `4891442` (JSON) y `d76610e` (memoria).
- **El título de la tabla es provisional.**
- **Pendiente (Jorge, 18 oct):** no le gusta cómo queda con los «…»; pensará cómo cambiarlo.
- **Texto de la 5.2: TERMINADO (18 oct).**

#### PROPUESTA DE TABLA (18 oct, aprobada por Jorge): acumulada, por etapas del flujo

**El problema de la tabla anterior.** Mezclaba dos tipos de cifra:
- **condicionadas, «sobre los detectados»:** tipo correcto 83,5 % y jugador 86,5 %;
- **acumuladas, «sobre todos los golpes reales»:** 80,1 % y 75,5 %.

La idea de Jorge es más clara: etapa 1, luego 1 y 2, y así hasta todo. Las etapas 2 (jugadores) y 3
(pelota) no tienen anotación en la que comprobarse; sirven a la 4 (asignación). Por eso los puntos de
control son 1 → 1+4 → 1+4+5.

**La tabla** (calculada el 18 oct desde `chain_cv_hits.csv`; las cifras finales coinciden con las del
JSON):

| Golpes que el sistema… | % | Intervalo 95 % |
|---|---|---|
| *Los 11 torneos (2210 golpes de las cuatro clases)* | | |
| detecta | 96,0 | 94,9-97,0 |
| detecta y clasifica bien | 80,1 | 77,4-82,6 |
| *VIGO, con el jugador anotado (286 golpes)* | | |
| detecta | 96,2 | 93,8-98,3 |
| detecta y asigna al jugador correcto | 84,6 | 76,3-90,9 |
| detecta, asigna al jugador correcto y clasifica bien | 75,5 | 66,9-82,2 |

**Detalles:**
- **Falta añadir al script de los intervalos:** las cifras de 96,0, 96,2 y 84,6 y sus intervalos no
  están en `confidence_intervals.json`. Hay que añadirlas a `scripts/evidence_confidence_intervals.py`;
  no hace falta GPU y las cifras actuales no cambian.
- **Pareja:** fuera de la 5.2, porque ya está en la 5.1.3.
- **Falsas alarmas, en la prosa:** 64 de las 2262 detecciones (2,8 %) no corresponden a ningún golpe
  real. Es la precisión de 0,972.
- **Por qué la detección no es la de la 5.1.1:** no es acumulación de errores, sino el protocolo. Aquí
  el torneo es nuevo.
- **Lo de «Other»:** en los 2377 golpes, el *recall* es 0,925; en los 2210 de las cuatro clases, se
  detecta el 96,0 %. Los golpes etiquetados como «Other» se detectan peor.

#### Detalle (explicación de trabajo, 18 oct; anterior a la propuesta nueva)

**Qué es y por qué importa.**
- **En la 5.1, cada componente se midió en las mejores condiciones para él:**
  - la asignación, con los instantes anotados;
  - el clasificador, con el golpeador que elige el voto en el instante anotado.
- **En uso real ninguno tiene esa ayuda:**
  - el audio decide cuándo;
  - el voto decide quién en el instante que encontró el audio, que puede estar algo desplazado o ser
    una falsa alarma;
  - el clasificador decide el tipo con el esqueleto del jugador que eligió el voto, que puede ser el
    equivocado.
- **Esta sección mide la cadena así,** que es lo que recibe de verdad quien usa el sistema.

**Cómo se mide.**
- **Un torneo fuera cada vez, como en la 5.1.4:** basta con recordarlo, sin volver a explicarlo.
  - En cada una de las 11 rondas se reentrenan sin ese torneo los dos modelos que aprenden de
    CVSPORTS: el detector de audio y el clasificador.
  - Los otros dos no lo necesitan: la pose usa el modelo preentrenado en COCO, y la pelota se entrenó
    con PadelTracker100.
  - Así, ningún modelo ha visto el torneo con el que se le mide.
  - El detector de audio de la 5.1.1 se midió con repartos al azar; aquí es más exigente, porque el
    torneo entero es nuevo.
- **Referencias de lo que es correcto:**
  - los 2377 golpes anotados de CVSPORTS, para el cuándo;
  - el etiquetado propio, para el tipo;
  - el jugador anotado por el paper, solo en VIGO, para el quién.
- **Emparejamiento:** un golpe detectado se empareja con uno real si está a menos de 250 ms, igual que
  en la 5.1.1.
- **Qué mide cada cifra:**
  - **Detección (F1):** sobre los 2377 golpes reales.
  - **Tipo correcto:** sobre los golpes detectados de las cuatro clases. Un golpe detectado sin
    jugador se queda sin clasificar y cuenta como fallo.
  - **Detectado y con el tipo correcto:** sobre todos los golpes reales de las cuatro clases; uno que
    no se detecta cuenta como fallo. Es la cifra que resume cuándo y qué en los 11 torneos.
  - **Jugador y pareja correctos (VIGO):** sobre los golpes detectados que tienen jugador anotado.
  - **Todo correcto (VIGO):** detectado, con el jugador y el tipo correctos, sobre los 286 golpes de VIGO
    de las cuatro clases con jugador anotado. Es la cifra más exigente.
- **Intervalos:** *bootstrap* por puntos, como en la introducción.

**Lectura.**
1. **La detección baja de 0,958 a 0,948, y la bajada está toda en el *recall*.**
   - El *recall* pasa de 0,945 a 0,925, y la precisión es la misma (0,972 en los dos casos). En un
     torneo nuevo, el detector no se inventa más golpes, pero se le escapan más.
   - Por torneo, el *recall* más bajo es el de Finlandia (0,853) y el de Bruselas (0,860). Malmö es el
     único con la precisión baja (0,886).
2. **Los errores se encadenan.** De cada 100 golpes reales de VIGO:
   - 96,2 se detectan;
   - 84,6 se detectan y van al jugador correcto;
   - 75,5 tienen además el tipo correcto.

   Cada paso pierde algo.
3. **El jugador equivocado le da al clasificador el esqueleto de otro.** En VIGO, el tipo acierta el
   89,3 % cuando el jugador es correcto, frente al 85,1 % sobre todos los detectados de VIGO. Hay que
   comparar dentro de VIGO, no con el 83,5 % de los 11 torneos.
4. **La asignación casi no empeora con los instantes del audio:** 86,5 % aquí frente a 87,5 % con los
   anotados.
   - El instante que detecta el audio basta para el voto.
   - Decirlo como «apenas cambia», no como una cifra exacta: los denominadores son distintos (aquí, los
     detectados con jugador anotado; allí, los 319) y los intervalos se solapan.
5. **Tipo: 83,5 % aquí frente a 82,7 % en la 5.1.4.**
   - No sacar conclusiones: los denominadores son distintos y los intervalos se solapan.
   - Como mucho, «se mantiene».
6. **Por torneo (opcional):** «detectado y con el tipo correcto» va del 67,4 % de Bruselas al 92,3 % de
   Malmö.

**Lo que no va.**
- El paper.
- La matriz de confusión de la cadena: repetiría la de la 5.1.4.
- Que en la cadena no se recupera el equipo por alternancia: es despreciable.

**Sugerencias de tabla y figura.**
- **Tabla de la cadena: sí, es la base.** Con el mismo formato que la de la asignación:
  - columnas Medida | Valor | Intervalo 95 %;
  - dos encabezados de grupo, «Los 11 torneos (2377 golpes)» y «VIGO, con el jugador anotado (286
    golpes)»;
  - filas en el orden de la cadena: detección (F1), tipo correcto, detectado y con el tipo correcto,
    jugador correcto, pareja correcta y todo correcto.
- **Embudo de VIGO:** en la prosa (96,2 → 84,6 → 75,5). Una figura de barras es opcional, porque la
  frase lo cuenta igual.
- **Por torneo:** como mucho, el rango en la prosa; sin tabla.

**Datos de cada torneo en la cadena** (`params.per_tournament_video`):

| Torneo | Golpes | Precisión | *Recall* | F1 | Tipo | Detectado y con el tipo |
|---|---|---|---|---|---|---|
| VIGO | 315 | 0,958 | 0,946 | 0,952 | 0,851 | 0,816 |
| Bruselas | 250 | 0,991 | 0,860 | 0,921 | 0,754 | 0,674 |
| Valladolid | 114 | 0,990 | 0,895 | 0,940 | 0,842 | 0,794 |
| Valencia | 104 | 1,000 | 0,952 | 0,975 | 0,851 | 0,816 |
| Málaga | 201 | 0,928 | 0,960 | 0,944 | 0,856 | 0,856 |
| Finlandia | 320 | 0,989 | 0,853 | 0,916 | 0,790 | 0,733 |
| Madrid | 296 | 0,979 | 0,949 | 0,964 | 0,825 | 0,810 |
| Alemania | 227 | 1,000 | 0,969 | 0,984 | 0,857 | 0,845 |
| Ámsterdam | 190 | 0,983 | 0,926 | 0,954 | 0,801 | 0,778 |
| Menorca | 187 | 1,000 | 0,952 | 0,975 | 0,859 | 0,831 |
| Malmö | 173 | 0,886 | 0,942 | 0,913 | 0,941 | 0,923 |

**Cómo se mide:**
- Un torneo fuera cada vez (el porqué ya se dio en el tipo de golpe, en la 5.1), reentrenando en cada
  ronda el audio y el clasificador sin ese torneo.
- La pose usa el modelo preentrenado en COCO, y la pelota está entrenada con PadelTracker100, fuera
  de CVSPORTS.
- Los intervalos remuestrean puntos.

**Cifras** (`chain_cv` + `confidence_intervals`):

| Medida | Valor | IC 95 % | Sobre qué golpes |
|---|---|---|---|
| Detección (F1) | 0,948 | 0,939-0,955 | los 2377 golpes reales (11 torneos) |
| Tipo correcto | 83,5 % | 81,1-85,7 | los detectados de las cuatro clases |
| Detectado y con el tipo correcto | 80,1 % | 77,4-82,6 | los reales de las cuatro clases |
| Jugador correcto (VIGO) | 86,5 % | 77,7-92,6 | los detectados con jugador anotado |
| Pareja correcta (VIGO) | 92,0 % | 84,1-97,3 | ídem |
| Detectado, jugador y tipo correctos (VIGO) | 75,5 % | 66,9-82,2 | los 286 golpes de VIGO de las cuatro clases con jugador anotado |

**Detalles de las cifras:**
- **Detección, desglose:** precisión 0,972 y *recall* 0,925. El macro-F1 del tipo es 0,863.
- **Denominadores:** salen de `padel_ml.chain_eval.summarize`.
  - Son 286 golpes y no 319 porque aquí solo entran los de las cuatro clases del etiquetado propio.
  - VIGO tiene 315 golpes reales en la cadena.
- **Embudo de VIGO** (`params.funnel_vigo_percent`). De cada 100 golpes reales:
  - 96,2 se detectan;
  - 84,6 se detectan y van al jugador correcto;
  - 75,5 tienen además el tipo correcto.

**Cómo interpretarlas:**
- **Por qué bajan: los errores se encadenan.** Un golpe no detectado no se puede asignar, y uno
  asignado al jugador equivocado le da al clasificador el esqueleto de otro.
  - Dato que lo apoya, en VIGO (`per_tournament_video.20230528_VIGO`): el tipo acierta el 89,3 %
    cuando el jugador es correcto, frente al 85,1 % sobre todos los detectados de VIGO.
  - Hay que comparar 89,3 con 85,1, los dos de VIGO, y no con el 83,5 de los 11 torneos.
- **Detección: 0,948 en la cadena frente a 0,958 en la 5.1.** Son protocolos distintos: aquí el
  torneo entero es nuevo, y allí el reparto es aleatorio por puntos. Es coherente que baje.
- **Tipo: 83,5 % en la cadena frente a 82,7 % en la 5.1.** No sacar conclusiones:
  - los denominadores son distintos (los detectados, con el jugador que elige el sistema, frente a
    los 2186 con jugador asignado);
  - los intervalos se solapan.
- **Por torneo:** también están los resultados de la cadena para cada torneo
  (`params.per_tournament_video`).

**Candidatos por decidir:**
- Tabla de la cadena: **es la base**, sí entra.
- Figura del embudo de VIGO, con un script nuevo.
- Cadena por torneo: probablemente no.

### 5.3 Coste de procesado

**Qué cuenta:** cuánto tarda en procesarse un punto, en qué se va el tiempo y en qué máquina se
midió.

**Cifras** (`processing_time.json`, commit `baec97a`):
- **Punto medido:** VIGO_11 (20230528_VIGO_11): 1608 fotogramas, 1920×1080 a 25 fps, 64,3 s de vídeo.
- **Qué se cronometra:** `analyze_rally` completo, sin contar una primera pasada de calentamiento.
- **Qué no se cuenta:** el vídeo anotado, que es una segunda pasada opcional.

| Etapa | Tiempo | Parte del total |
|---|---|---|
| Audio: detección del golpe | 0,2 s (0,17) | 0,3 % |
| Pose de los jugadores | 27,1 s | 40,2 % |
| Pelota | 26,3 s | 39,0 % |
| Resto: lectura del vídeo, identidad J1-J4, limpieza de la pelota, voto y clasificador | 13,8 s | 20,5 % |
| Total | 67,5 s | |

- **Factor: 1,05 veces el tiempo real** (1,049). Un minuto de vídeo se procesa en algo más de un
  minuto.
- **Cautela con los porcentajes.**
  - `sistema.md` dice 0,2 % y 39,1 %, calculados con los tiempos sin redondear.
  - Desde el JSON salen 0,3 % y 39,0 %, que son los que cuadran con los segundos que imprime la tabla.
  - Usar los del JSON, o escribir «menos del 1 %» para el audio.
- **Lectura:** el audio resuelve la pregunta más difícil (cuándo) y apenas cuesta nada. El gasto está
  en la visión, a partes casi iguales entre la pose y la pelota.

**Tabla del equipo** (consultado en la máquina el 17 oct 2026):

| Componente | Valor | De dónde sale (para el script) |
|---|---|---|
| Tarjeta gráfica | NVIDIA GeForce RTX 3090, 24 GB | `nvidia-smi --query-gpu=name,memory.total` (24 576 MiB). Debe coincidir con `processing_time.params.gpu` |
| Procesador | AMD Ryzen 9 5900X, 12 núcleos y 24 hilos | `/proc/cpuinfo` (`model name`, `cpu cores`) y `nproc` |
| Memoria | 32 GB | Windows: `Win32_PhysicalMemory` a través de `powershell.exe` (34 359 738 368 bytes) |
| Sistema | Windows 11 Pro con WSL2, Ubuntu 26.04 LTS | `Win32_OperatingSystem.Caption` y `/etc/os-release` |
| Software | Python 3.11.15, PyTorch 2.13.0, CUDA 13.0 | `platform.python_version()`, `torch.__version__` y `torch.version.cuda`. `uv.lock` no ha cambiado desde la medida |

La memoria se lee de Windows y **no** de WSL: WSL ve 19 GiB, que es lo que Windows le asigna.

**Candidatos por decidir:**
- Tabla del coste: **es la base**, sí entra.
- Una fila «Duración del vídeo: 64,3 s» bajo el total, o ese dato solo en la prosa.
- Tabla del equipo: sí entra (decidido por Jorge).

## Pendiente fuera de este capítulo

- **REVISIÓN DE AUTORÍA EN EL CAPÍTULO 4 (feedback del padre de Jorge, 18 oct). Hacerla antes de
  pasar el borrador al tutor.**
  - **El problema:** se cita mucho a [1] (Decorte) y no queda claro qué es de este trabajo, qué se
    reutiliza y de dónde sale cada cosa.
    - Ejemplo: «un \textit{dataset} previamente anotado~\cite{decorte2024padel}». ¿Anotado por quién?
      ¿Qué es?
    - Preguntas que se hizo: ¿el audio es copiar y pegar lo suyo? ¿Lo de normalizar las distancias es
      nuestro?
  - **Material de partida:** `docs/sistema.md`, «Modelos: qué es de quién», y lo que se sabe:

    | Componente | De terceros | De este trabajo |
    |---|---|---|
    | Audio | Arquitectura de Decorte (adaptación de SED-net); datos e instantes de CVSPORTS, anotados por Decorte et al. | Reimplementación, entrenamiento, umbral y evaluación |
    | Pose | YOLO26n-pose con los pesos COCO (Ultralytics), tal cual; ByteTrack | Configuración de ByteTrack, máscara de la pista; identidad J1-J4 (3 primeros segundos, citada a Decorte; comprobar qué es nuestro) |
    | Pelota | Idea de TrackNet; datos de PadelTracker100 | Red propia simplificada (U-Net reducida), entrenada desde cero en pádel |
    | Asignación | Voto ponderado en varios fotogramas (Decorte) | Normalizar por la altura (remates) y la ventana de ±4 |
    | Tipo de golpe | BST (Chang) como base | Etiquetado de 2377 golpes, adaptación (pose de un solo jugador, ventanas, regla del saque), entrenamiento y evaluación |
    | Pista | Seis clics por torneo, como en el paper (ADR-0018) | Herramienta de marcado propia |
    | Visor | — | Entero |

  - **CUÁNDO:** al FINAL, cuando estén todos los capítulos. Primero se cambia el capítulo 4 y después
    se hace un repaso general, antes de los anexos (decisión de Jorge, 18 oct).
  - **Las seis categorías de Jorge** (18 oct), para que el lector sepa de cada pieza si se toma, se
    adapta o es propia:
    1. **Replicar tal cual, entrenado por Jorge** (no sus modelos): **audio**, reimplementado sin
       cambios a partir de Decorte, con su *dataset*.
    2. **Usar tal cual el modelo de otro:** **YOLO26n-pose** (pesos COCO).
    3. **Basarse y añadir funcionalidades:** **asignación**, el voto de Decorte más la normalización por
       la altura. La ventana es de ±4 fotogramas; la de Decorte, de ±6.
    4. **Basarse cambiando capas o variables:** **pelota**, la idea de TrackNet con una U-Net reducida.
    5. **Usar de referencia y adaptar a otra disciplina:** **BST**, del bádminton al pádel.
    6. **Datos de otros, entrenado por Jorge:** el **audio** (CVSPORTS) y la **pelota**
       (PadelTracker100).
  - **Lo que no se cita:**
    - **La máscara de la pista:** idea propia, y no se parece a la de Decorte (decisión de Jorge).
    - **«La ventana»:** OJO, la ventana adaptativa del clasificador SÍ es de BST (§3.1 y
      suplementario C.2). Aunque no lleve cita propia, el texto tiene que dejar claro que se toma de
      BST, por ejemplo por la frase «el clasificador parte de BST». Tampoco es nuestra la idea del
      voto con varios fotogramas, que es de Decorte; sí lo es el ancho de ±4.
  - **Cómo hacerla:** recorrer cada cita a `decorte2024padel` y cada «se utiliza» del capítulo 4.
    Que cada frase deje claro si algo se toma (de quién), se adapta (qué cambia) o es propio.
  - **Numeración de las citas:** es por orden de aparición (estilo IEEE de biblatex). Si la
    introducción cita antes a otros, Decorte deja de ser [1], y se renumera solo al compilar.

- **Notas `%` de Jorge en el capítulo 4: RESUELTAS (18 oct).**
  - `tab:deteccion`, sin columna de tiempo; Jorge añade la frase de que el audio apenas suma tiempo.
  - `tab:red-audio`, subida tras el párrafo del *dataset*: ahora cae en la página de la 4.2.1.
  - `tab:pose-resolucion`, con el tiempo adicional respecto a 640 px (+4,2, +15,4 y +18,2 %).

  Las notas eran:
  1. **`tab:deteccion`:** quitar la columna de tiempo y decir en el texto que el audio solo añade un
     pequeño porcentaje de cómputo. La pose y la pelota se calculan siempre, así que no son
     alternativas.
  2. **`tab:red-audio`:** la tabla aparece en la sección de jugadores; colocarla más arriba.
  3. **`tab:pose-resolucion`:** dar el tiempo como porcentaje sobre 640 px, no en ms, porque el tiempo
     depende del equipo.
- **Pasada de conectores: HECHA en los capítulos 5 y 4 (18 oct).** En el capítulo 4, 16 cambios: la
  4.1, de la 4.2.1 a la 4.2.4, el principio de la 4.2.5 y la 4.3.
- **Pasada de conectores al estilo Miriam** (Jorge, 18 oct), al terminar el capítulo 5:
  - en todo el capítulo 5;
  - en todo el capítulo 4, donde solo se hizo en la 4.2.5 (tipo de golpe).

  La guía está en `notas-tipo-de-golpe.md`.

- **Limitaciones de `sistema.md`:**
  - quién golpea se mide en un solo torneo;
  - los ajustes de la asignación se eligieron sobre los mismos golpes;
  - un único etiquetador;
  - el jugador de cada ejemplo de entrenamiento lo elige el voto;
  - la pelota se mide en otro partido de la misma pista;
  - todo son retransmisiones del World Padel Tour.

  Algunas ya salen como cautelas en la 5.1 y la 5.2; falta decidir si las demás van en Conclusiones.
- **Conclusiones:** la limitación de los fps y el trabajo futuro.

## Descartado (no entra en la memoria)

Notas que venían del capítulo 4.

- **Tamaño de entrada de la pose:** ya está como tabla en la 4.2.2 (`tab:pose-resolucion`).
- **YOLO26n frente a YOLO26m, umbral de la pose y ByteTrack:**
  - lo cubre la frase de los parámetros fijados empíricamente;
  - no tienen script ni JSON, solo comentarios en `scripts/extract_rally_features.py` y una prueba
    antigua en `docs/experiments.md`;
  - si el tribunal pregunta por ByteTrack, no afirmar que se aprovechan las detecciones de baja
    confianza, porque el detector ya corta en 0,25.
- **Homografía y pista detectada automáticamente** (ADR-0018):
  - el capítulo 4 solo dice que la alternativa automática no era fiable;
  - las cifras del ADR no se citan sin comprobarlas: reproyección de unos 0,11-0,15 m en 6 torneos,
    deriva en 3 y fallo en 2;
  - el residuo de 0,113 m mide consistencia, no exactitud.

# Notas para el capítulo 1 (Introducción)

Referencia de trabajo mientras se escribe el capítulo; se borra al terminar la memoria.
No es texto de la memoria: es la estructura acordada, las cifras comprobadas y lo que va en cada
parte.

## Estado

- **19 oct:** se empieza el capítulo; ya están cerrados del 2 al 6 (el 2, comprobado y compilado sin
  avisos tras aplicar la revisión de redacción y conectores).
- **Método:** como en el resto, primero el contenido explicado, después Jorge redacta, luego la
  revisión de redacción y por último los conectores. Sección a sección.

## Estructura (ya fijada en el README y validada por el tutor)

1. **1.1 Motivación y antecedentes** (`sec:motivacion`)
2. **1.2 Objetivos generales y específicos** (`sec:objetivos`)
3. **1.3 Metodología del trabajo** (`sec:metodologia`). La rúbrica le da peso: «metodología apropiada
   y bien aplicada» es el 30 % de la calidad del trabajo.

**Opcional, al final de la 1.3:** un párrafo con la estructura de la memoria, sin sección propia.
Decide Jorge al llegar.

## Lo que ha quedado ANTICUADO en las `% IDEAS` del `.tex`

- **Cartes Domínguez y Padel AI:** fuera, como en el capítulo 2.
- **«ni herramientas abiertas»:** es falso, porque existe Padex (código abierto, reglas sobre la pose y
  sin evaluación). El hueco se dice solo de los **trabajos publicados**, como en la 2.4.
- **«Listón: que funcione lo bastante bien»:** fuera. Es un juicio subjetivo, y lo decide quien use la
  herramienta.
- **«Para quién: clubes»:** son **clubes, equipos e incluso profesionales**, como en la 6.1.
- **La cámara:** no se presenta como requisito (Jorge: «eso es mentira»). Basta con decir que se
  evalúa con grabaciones de retransmisión, que son los datos anotados de que se dispone.
- **El alcance:** tiene que coincidir con el trabajo futuro de la 6.2. Queda fuera:
  - más tipos de golpe (el globo y la volea, no «bandeja, víbora y dejada»);
  - otras cámaras;
  - los partidos completos;
  - el desenlace del punto;
  - y, si se decide al final, los botes y la trayectoria.

## Cifras comprobadas en el PDF (FIP World Padel Report 2025, `fip2025report`)

- **Jugadores:** más de 35 millones de aficionados, y más de la mitad juega al menos una vez por
  semana (pág. 14 del informe).
- **Pistas:** más de 77 000 a finales de junio de 2025. En los últimos 18 meses se han sumado más de
  14 355, un 22,8 % más, lo que equivale al 15,2 % anual (pág. 18).
- **Instalaciones:** más de 24 600 clubes e instalaciones (pág. 16).
- **Países:** se juega en 150 países y 20 territorios dependientes (pág. 16).
- **Sobre la fuente:** es un informe de la federación internacional, no un artículo revisado por
  pares. Es la fuente primaria de la cifra, y ya estaba acordado usarlo para el crecimiento.

## 1.1 DECIDIDO (Jorge, 19 oct): esquema de Miriam en tres párrafos, sin antecedentes ni hueco

1. **Contexto:** la popularidad con las cifras de la FIP, el crecimiento de la investigación
   (Martín-Miguel) y la visión artificial (`\cref{sec:arte-padel}`).
2. **Reto:** se abre con el problema descrito. Las estadísticas por jugador exigen saber, de cada
   golpe, cuándo se da, quién lo da y de qué tipo es. Las dificultades:
   - son tres problemas encadenados, y un error en uno pasa al siguiente;
   - los golpes son rápidos y breves y se diferencian en detalles sutiles;
   - con la cámara de fondo, los jugadores del otro lado se ven pequeños.

   El reto da el argumento de la complejidad técnica, que es el 70 % de la calidad en la rúbrica.
3. **«Por todo ello, este trabajo se centra en…»:** del vídeo de un punto, cuándo, quién y de qué
   tipo es cada golpe, y una herramienta para clubes, equipos e incluso profesionales. Cierra con
   `fig:problema`.

- **El hueco NO va** («ningún trabajo publicado…», las aplicaciones comerciales): ya está en la 2.4.
- **Novillo y Decorte NO van en la 1.1:** son el punto de partida de la **1.3 Metodología**. Se parte
  del sistema de los tutores, se reimplementa el cuándo y el quién de Decorte y se valida
  comparándose con ellos, y después se añade la pieza nueva. Así la autoría queda clara en el
  capítulo 1 sin repetir la 2.1.
- **Título:** se mantiene «Motivación y antecedentes»; el contexto hace de antecedentes.

### Resumen detallado para redactar la 1.1 (19 oct)

**Martín-Miguel, comprobado en el PDF** (`papers_tfg/JHK-89-168640.pdf`):
- el aumento de jugadores ha traído más publicaciones;
- el 77,7 % de los artículos son de 2020 en adelante;
- el análisis del rendimiento consiste en analizar, registrar y evaluar las acciones de juego de los
  jugadores en partidos reales, para que después las usen los entrenadores (pág. 214);
- de 27 estudios, 23 tratan las acciones de juego: es el tema más estudiado (pág. 215);
- **no habla de vídeo:** no decir que sus estudios usan vídeo.

**Párrafo 1, el contexto** (dos citas):
1. Una afirmación sobre el crecimiento del pádel, sin comparativos que la fuente no respalde
   («el deporte que más crece»).
2. Las cifras de la FIP (`fip2025report`), dos como máximo: más de 35 millones de aficionados; más
   de 77 000 pistas en junio de 2025, un 22,8 % más en 18 meses.
3. La investigación crece con él (`martinmiguel2023performance`). El análisis del rendimiento
   registra las acciones de juego en partidos reales para los entrenadores, y es el tema más
   estudiado (23 de 27).
4. Registrar cada acción de un partido es lo que la visión artificial puede automatizar a partir del
   vídeo. En pádel ya se obtienen las posiciones, la pelota y el instante y el autor de cada golpe
   (`\cref{sec:arte-padel}`, sin cita).

**Párrafo 2, el reto** (sin citas):
1. El problema: las estadísticas por jugador exigen saber, de cada golpe, cuándo se produce, quién
   lo da y de qué tipo es.
2. Los tres pasos están encadenados: con el jugador equivocado se clasifica el gesto de otro. Sin
   cifras.
3. Los golpes son rápidos y breves y se diferencian en detalles; por ejemplo, la derecha y el revés
   se distinguen por el lado del cuerpo.
4. La cámara fija al fondo: los jugadores del otro lado ocupan pocos píxeles. Opcional: los cercanos
   se ven de espaldas.

**Párrafo 3, «Por todo ello…»** (sin citas, una o dos frases):
- un sistema que, del vídeo de un punto, obtiene cuándo, quién y de qué tipo es cada golpe;
- y lo lleva a una herramienta de análisis para clubes, equipos e incluso profesionales;
- el detalle va en la 1.2.

**No va:** el hueco (2.4), Novillo y Decorte (1.3), cifras de resultados, «fiable» ni «preciso».

**La figura `fig:problema`: FUERA de la introducción** (recomendación aceptada si Jorge no dice lo
contrario).
- Miriam no la pone, y las introducciones de un TFG no suelen llevarla.
- La salida del sistema ya se ve en el capítulo 4 (`fig:flujo`) y en el anexo del visor.
- El punto que muestra lo vieron los modelos al entrenar, y el tribunal podría preguntarlo.
- Se conservan el PDF y su script (`scripts/build_problem_figure.py`) para las diapositivas de la
  defensa.

## 1.1: propuesta inicial (superada)

Tres párrafos cortos, sin repetir el estado del arte, que se referencia con `\cref`.

1. **El contexto.**
   - El pádel crece: dos cifras de la FIP como máximo (jugadores y pistas, o el crecimiento).
   - Analizar el juego, es decir, qué golpes da cada jugador, sirve para mejorar el rendimiento.
   - En los trabajos de rendimiento, el tipo de cada golpe se registra a mano (`\cref{sec:arte-tipo}`).
     Ramón-Llin va citado allí y no se repite la cita.
2. **Los antecedentes.**
   - Este trabajo parte del de Novillo et al., los tutores: extraen del vídeo la pista, los jugadores
     y la pelota y representan el juego en 2D. Dejan el tipo de golpe como trabajo futuro (§4, pág.
     119, comprobado), y este trabajo lo aborda.
   - Para el cuándo y el quién, se parte del método de Decorte et al., que se reimplementa. Así, el
     lector sabe desde el principio qué se toma y qué es propio, que es el problema de autoría que
     señaló el padre de Jorge.
3. **El problema.**
   - Hasta donde se ha podido comprobar, no hay ningún trabajo publicado que clasifique el tipo de
     golpe en pádel a partir de vídeo; las aplicaciones comerciales no publican ni su método ni su
     precisión (`\cref{sec:arte-tipo}`). Una frase, sin repetir la 2.4.
   - Lo que se aborda: del vídeo de un punto, cuándo, quién y de qué tipo es cada golpe.
   - Cierre con la figura `fig:problema`.

### Primer párrafo con el esquema de la introducción de Novillo et al. (Jorge, 19 oct)

Jorge quiere que el contexto siga el esquema de la introducción de Novillo:
1. la popularidad del pádel;
2. que con ella ha crecido la investigación;
3. que la visión artificial y la IA entran en esos estudios.

Se toma **solo el esquema**: ni una frase suya, ni traducida. Además, se cambian sus adjetivos
(«immense popularity», «revolutionized») por cifras, que no son juicios.

- **Sus referencias, comprobadas en su PDF:**
  - [3] García-Giménez 2022 (IJERPH, revisión de alcance);
  - [10] Martín-Miguel 2023 (J. Hum. Kinet., revisión sistemática);
  - [12] Priego Quesada 2013 (JHSE);
  - [6] Javadiha 2021;
  - [13] Santiago 2010 y [14] Shih 2017, encuestas de visión en el deporte.
- **Se usa Martín-Miguel et al. (`martinmiguel2023performance`, añadida al `.bib`):**
  - Comprobado en la web de la revista: «The increase in the number of players has also led to a
    greater number of scientific publications in this field».
  - Revisan 27 estudios, todos posteriores a 2018, en cuatro áreas: aspectos temporales, acciones
    de juego, desplazamientos en la pista y marcador.
  - Con una revisión basta. García-Giménez no aporta nada distinto; Priego Quesada (2013) y las
    encuestas generales tampoco hacen falta.
  - **Pendiente:** el PDF completo (es de acceso abierto) en `papers_tfg/`, para la defensa.
- **La visión artificial en pádel:** sin cita nueva; se remite a `\cref{sec:arte-padel}`.
- **Cuidado:** que los estudios de rendimiento registren los golpes «a mano» solo está comprobado
  para Ramón-Llin, en la 2.4. En la introducción no se generaliza: se dice que hay que registrar
  cada acción del juego, y que la visión artificial puede automatizarlo.

**Por decidir con Jorge:**
- qué cifras de la FIP;
- si Decorte va en la 1.1;
- la figura `problema.pdf`, que puede cambiar.

## RESUMEN: CRITERIO DE JORGE (20 oct), sustituye a la propuesta de abajo

**Un resumen de TFG no es un *abstract* de artículo:**
- cero cifras, cero citas, cero nombres de autores, cero detalles técnicos;
- solo qué se hace, para qué y qué presenta el documento, en lenguaje llano, para que se lea y se
  entienda.

Es el esquema de Miriam (cuatro párrafos):
1. **El objetivo principal**, la herramienta para clubes, equipos y profesionales, y el contexto.
2. **Lo que hace falta:** un sistema que, de cada golpe, sepa cuándo, quién y de qué tipo es,
   combinando el sonido con el movimiento de los jugadores y de la pelota.
3. **Qué presenta el documento:** el etiquetado propio, cómo se resuelve cada parte y la evaluación
   con torneos no vistos.
4. **Cierre:** el visor y, si se quiere, lo que abre el trabajo.

Sin la frase del hueco. Palabras clave: unas 8-10, como Miriam.

## RESUMEN Y PALABRAS CLAVE (20 oct, propuesta SUPERADA)

- **Normativa (art. 16):** resumen de UNA página y palabras clave, obligatorios. El inglés no se
  exige (`README.md`: «se decide al escribir el resumen»).
- **Plantilla:** `pages/resumen.tex`, vacío salvo «Palabras clave:». Va en las páginas romanas, así
  que no cuenta para las 40-60.
- **Estructura propuesta** (unas 300 palabras con 12 pt, cuatro párrafos):
  1. **Contexto y problema**, en 2 frases. El hueco en una frase, si Jorge quiere: en el resumen es
     lo habitual, aunque no lo repitió en la 1.1.
  2. **Qué se hace:**
     - el cuándo: audio, reimplementando a Decorte;
     - el quién: pose y pelota, con el voto más la normalización por la altura;
     - el tipo: cuatro clases, con el clasificador adaptado de BST sobre la pose y la pelota;
     - el etiquetado propio: 2377 golpes de 11 torneos;
     - el visor.
  3. **Resultados**, de 3 a 4 cifras como mucho, todas del capítulo 5:
     - detección: F1 0,958;
     - tipo: 82,7 % de *accuracy* y 0,859 de macro-F1, dejando fuera un torneo;
     - sistema completo: 80,1 % detectados y bien clasificados en los 11 torneos; 75,5 % totalmente
       correctos en VIGO;
     - coste: 1,05 veces la duración del vídeo;
     - (asignación: 87,5 % del jugador en los 319 golpes de VIGO).
  4. **Cierre** en una frase, sin juicios: lo que permite el visor, o el trabajo futuro.
- **Palabras clave**, de 5 a 6: pádel, visión artificial, detección de eventos sonoros, estimación
  de la postura, clasificación de golpes, *transformer*.

## 1.2 Objetivos: propuesta (19 oct, pendiente de que Jorge la confirme)

Jorge plantea: el general es la web y los específicos son el sistema; quizá también el etiquetado; y
el listón, ¿un número, «hasta estar satisfecho» o nada?

- **Un objetivo general,** como Miriam y como ya dice la 6.1 que escribió Jorge («El principal
  objetivo… una herramienta… Para ello, fue necesario… un sistema»): que clubes, equipos e incluso
  profesionales dispongan de una herramienta para analizar qué golpes da cada jugador a partir del
  vídeo. Sustituye a los «dos objetivos» de `notas-conclusiones.md`.
- **Específicos, en prosa** («En primer lugar… Para ello… Además… Por último»):
  1. detectar el instante de cada golpe;
  2. identificar al jugador que lo da;
  3. clasificar su tipo en cuatro clases. Para ello hay que etiquetar el tipo, que el conjunto de
     partida no incluye: el etiquetado entra como **medio** dentro de este objetivo, no como objetivo
     propio. Jorge dijo el 18 oct que no es un objetivo («dificultad 0»). Nada de «no existe ningún
     conjunto»: PadelTracker100 tiene etiquetas de tipo y no se menciona;
  4. unirlos en un sistema: del vídeo de un punto, por cada golpe, el instante, el jugador y el tipo;
  5. el visor: las estadísticas por jugador y el salto a cada golpe;
  6. opcional: medir cada componente y el sistema completo, sin el protocolo.
- **El cómo NO va:** que el instante salga del audio es un hallazgo (la 6.1 lo concluye), no un
  objetivo de partida.
- **El listón:**
  - ni un número inventado: un umbral fijado después de ver los resultados se nota, y el tribunal lo
    puede preguntar;
  - ni «hasta estar satisfecho», que es subjetivo;
  - la respuesta son las cifras del capítulo 5, y decide el usuario (lección de Ferrovial).
- **El alcance:** dos frases al final, sin repetir la 6.2. Puntos ya recortados, de retransmisiones
  con cámara fija al fondo, y cuatro tipos agrupados por gesto.
- **La extensión:** media página está bien; el peso del capítulo está en la 1.3.

## 1.3, versión de Jorge (19 oct): cambios sobre el borrador

- **Quitado por Jorge («no me gusta y fin»):**
  - el ejemplo del audio y la vuelta a la literatura cuando una solución no basta;
  - las frases de Novillo, Decorte y BST.

  Así que la autoría NO queda en el capítulo 1: la lleva entera la revisión del capítulo 4.
- **Añadido por Jorge:**
  - el formato ADR con su estructura;
  - la bitácora con sus 30 entradas y el patrón de cinco pasos, en lista numerada;
  - dos FIGURAS (pendientes, después del texto): un recuadro que muestre cómo se ve el `.md`
    renderizado, al estilo de GitHub, una para un ADR y otra para una entrada de la bitácora. Es
    como el recuadro de código de Miriam, pero en markdown.
- **La estructura real de los ADR:**
  - **Estado:** aceptada, aceptada en parte o superada, con su fecha. No hay «rechazada».
  - Después, **Contexto**, **Decisión** (9 de ellos dicen «Decisiones») y **Consecuencias**.
- **Cita del formato ADR:** su origen es una entrada de blog de Nygard (2011), que no es revisada por
  pares. Recomendación: sin cita; o, si Jorge la quiere, Tyree y Akerman (IEEE Software, 2005),
  comprobándola antes en el PDF.
- **Figura de la bitácora:** OJO con las cifras anteriores a la limpieza. La entrada «HITO: la
  resolución de la pelota» da un 93,73 % de equipo frente al 92,2 % del capítulo 5. Elegir una
  entrada cuyas cifras coincidan con las finales, o recortarlas.

## 1.3 DECIDIDO (Jorge, 19 oct): cómo se ha trabajado, SIN fases ni fechas

**Por qué no hay fases:** Miriam tenía un TFG con horario y tutor presente. Jorge ha trabajado con
libertad total; con el tutor solo habló para enseñarle la web. No se mencionan reuniones ni el
tutor, ni párrafo de estructura de la memoria (Jorge: no).

**Un párrafo por principio:**
1. **Incremental e iterativo:**
   - primero, una versión completa que funciona de principio a fin;
   - después se mejora cada componente sin que el sistema deje de funcionar («demo siempre verde»);
   - cada componente sigue el ciclo de investigar, implementar, medir y decidir, y si no basta, se
     vuelve a la literatura;
   - un solo ejemplo: el instante por imagen, que da paso al audio (`tab:deteccion`).
2. **Partir de lo publicado:**
   - el planteamiento de Novillo et al. (no su código);
   - se reimplementa el cuándo y el quién de Decorte et al. y se valida con sus resultados
     publicados ANTES de construir encima;
   - el clasificador parte de BST.

   Aquí queda clara la autoría desde el capítulo 1.
3. **Decisiones documentadas:** un ADR por decisión (contexto, decisión y consecuencias), 18 en
   total, que conservan los superados; y la bitácora de experimentos.
4. **Rigor de la medida:**
   - una sola función para la demo, la evaluación y el visor: lo medido es lo que se ejecuta;
   - una auditoría final que vuelve a medir cada cifra con la configuración final;
   - cada figura y tabla sale de un script.
5. **Prácticas de ingeniería,** en una frase: control de versiones, pruebas automáticas y
   comprobación estricta de estilo y de tipos.

**Herramientas:** NO van en la 1.3. La pila (Python, PyTorch, uv, etc.) va al anexo A
(Reproducibilidad), que está vacío. Sin cifras de resultados.

## 1.3: primera propuesta por fases (superada)

Jorge pasa la metodología de Miriam como referencia: cuatro fases (documentación, desarrollo de
software, simulaciones y análisis), una por párrafo, con qué se hizo en cada una.

**La historia real,** sacada de git (341 commits: julio 114, agosto 14, septiembre 83, octubre 130),
de los ADR y de `docs/experiments.md`:
- **Julio y agosto:**
  - el andamiaje, la pose (YOLO26), la pista automática (ADR-0001 y 0006), la API con cola
    (ADR-0007) y la web;
  - el golpe por la muñeca, después por la pelota (ADR-0012, *recall* del 15-52 %), después con dos
    modelos (ADR-0013) y después por fotograma (ADR-0014, *recall* de ~50 %).
- **Septiembre:**
  - giro al audio (ADR-0015): se replica a Decorte, con F1 0,93 y el equipo igual que el paper;
  - lectura completa de E2E-Spot, BST y Santra, y de ahí BST (ADR-0016);
  - el etiquetado de 2377 golpes;
  - un solo camino de código (`analyze_rally`).
- **Del 30 sep al 2 oct:** la limpieza. Se vuelve a medir cada cifra con la configuración exacta del
  sistema; se encontraron tres diferencias en la asignación.
- **Octubre:** la web estática (ADR-0017), la pista a mano (ADR-0018), las evidencias y la memoria.

**Cuatro fases, alineadas con los capítulos:**
1. **Investigación y prototipo** (capítulo 2):
   - los trabajos de pádel; se parte del PLANTEAMIENTO de Novillo et al., no de su código (que no se
     ha usado);
   - un primer prototipo completo de principio a fin;
   - con él se prueban las vías por imagen para el instante.
2. **La base, cuándo y quién** (4.2.1-4.2.4):
   - se reimplementa a Decorte et al.;
   - se valida comparándose con sus resultados publicados (5.1) antes de construir encima, porque un
     error en la base pasa a todo lo demás.
3. **El tipo de golpe** (capítulo 3 y 4.2.5): la lectura completa de los trabajos de clasificación,
   BST, el etiquetado propio, y entrenar y evaluar dejando fuera torneos.
4. **Integración, visor y evaluación final** (4.3 y capítulo 5):
   - una sola función para la demo, la evaluación y el visor, así que lo medido es lo que se ejecuta;
   - una auditoría final que vuelve a medir cada cifra;
   - cada figura y tabla, regenerable con un script.

**Transversal, un párrafo:**
- **El ciclo:** investigar, implementar, medir y decidir.
- **Cada decisión, en un ADR** (contexto, decisión y consecuencias). Son 18, y los superados se
  conservan porque explican el camino.
- **La bitácora de experimentos.**
- **«Demo siempre verde»:** el sistema funciona de principio a fin en todo momento.
- **Pruebas automáticas,** comprobación de estilo y de tipos estricta (ruff, mypy), y git con
  mensajes convencionales.

**Sin cifras de resultados** (van en el capítulo 5) y **sin fechas**, salvo que Jorge las quiera.
**Pendiente de preguntar:** las reuniones con los tutores; las «dos hipótesis del tutor» de la
bitácora salieron negativas.

## 1.2 y 1.3 (después)

- **1.2:** dos objetivos generales, la herramienta para clubes, equipos e incluso profesionales y el
  sistema que la hace posible (ver `notas-conclusiones.md`).
  - Específicos: el instante, el jugador, el tipo (cuatro clases) y el visor.
  - En prosa con conectores.
  - Los datos etiquetados NO son un objetivo.
  - El alcance va al final, en pocas frases.
- **1.3:** desarrollo por niveles, cada uno demostrable de principio a fin y probado antes del
  siguiente; decisiones documentadas en ADR (19); bitácora de experimentos; pruebas y tipado estricto;
  toda figura y tabla sale de un script. Concretar al llegar.

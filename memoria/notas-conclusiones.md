# Notas para el capítulo 6 (Conclusiones y trabajo futuro)

Referencia de trabajo mientras se escribe el capítulo; se borra al terminar la memoria.
No es texto de la memoria: es la estructura acordada y lo que va en cada parte.

## Estado

- **18 oct:** estructura acordada con Jorge: **6.1 Conclusiones y 6.2 Trabajo futuro**, sin sección
  de limitaciones.
- **Validado por Jorge (18 oct):** las cinco ideas de la 6.1 y las cuatro líneas de la 6.2. La
  cadena va como conclusión, no como trabajo futuro.
- **`pages/conclusiones.tex`:** ya tiene las dos secciones, con las etiquetas `sec:conclusiones` y
  `sec:trabajo-futuro`; todavía sin texto.
- **El objetivo, corregido por Jorge (18 oct).** No es «clasificar golpes». El objetivo es que el
  sistema sirva a **clubes, equipos e incluso profesionales**.
  - Clasificar el tipo es la pieza que faltaba para conseguirlo.
  - Es también lo que el trabajo de partida de sus tutores, Novillo et al. (`novillo2025padel`, §4,
    pág. 119), dejaba como trabajo futuro: «incorporating information about the types of shots made
    by players».
  - Las `% IDEAS` de la introducción dicen solo «clubes»: hay que ampliarlas a equipos y
    profesionales al escribirla.
- **DOS OBJETIVOS (Jorge, 18 oct, corregido).**
  1. **La herramienta de análisis** (el visor), para clubes, equipos e incluso profesionales: el
     objetivo final de Jorge. Su tutor la llamó «bonus».
  2. **El sistema que la hace posible,** igual de importante: del vídeo de un punto, saca por sí solo
     el instante, el jugador y el tipo de cada golpe. Combina el audio, la pose y la pelota, la
     asignación y la adaptación de BST. Ahí está la dificultad y el valor. Sin un sistema fiable no
     hay herramienta.
  - **Los datos etiquetados NO son un objetivo** («cualquiera puede etiquetar»). Fueron necesarios
    porque nadie los había hecho, y quedan como aportación reutilizable. Se mencionan en la idea 3,
    como lo que hizo posible entrenar y evaluar el clasificador. Las `% IDEAS` de la introducción
    («No son objetivos: los datos etiquetados») siguen siendo correctas.
  - **En las conclusiones:** dos frases al principio, la herramienta y después el sistema como
    condición para que tenga valor.
  - **Coherencia con la introducción:** los objetivos específicos de detectar el instante,
    identificar al jugador y clasificar el tipo son el sistema; el visor es la herramienta.
  - **«No existía» (para los datos):** con matiz, «hasta donde se ha podido comprobar». Confirmarlo
    en el estado del arte (hasta ahora solo hay sensores IMU y aplicaciones comerciales cerradas).
- **Cómo decir el objetivo sin pasarse (18 oct).** El objetivo es lo que se propuso, no lo que se
  demuestra.
  - **Objetivo:** que clubes, equipos e incluso profesionales dispongan de una herramienta de
    análisis.
  - **Conclusión (solo lo medido):** ya es utilizable para analizar puntos grabados con una cámara
    fija y elevada al fondo de la pista, como la de las retransmisiones. Los datos son de partidos
    profesionales: para equipos y profesionales con ese tipo de grabación, el respaldo es directo.
  - **Trabajo futuro:** llevarlo a las cámaras de cualquier club (otras alturas y posiciones), que es
    la línea 2. Lo que no se ha medido no se afirma.
  - **El entusiasmo del tutor** no es un argumento para la memoria; lo son los resultados del
    capítulo 5.
- **El sonido:** la idea es de Decorte et al., así que se presenta como algo que este trabajo
  *confirma* con una medida propia: imagen frente a audio en los mismos golpes (`tab:deteccion`).
  Con dos frases basta, citando a Decorte. Esto encaja con la revisión de autoría.

## Criterio (Jorge, 18 oct, con el ejemplo de Miriam)

- **Las conclusiones son del trabajo, no un repaso de resultados:** el análisis de las cifras ya está
  en el capítulo 5.
  - **Fuera:** recorrer cada objetivo con su cifra y repetir las del sistema completo («tres de cada
    cuatro golpes»).
  - **Dentro:** la valoración global con su utilidad («funciona lo bastante bien para usarse en clubes
    o para analizar puntos»).
- **Patrón de Miriam:**
  - empieza retomando el objetivo principal y por qué importa;
  - cada párrafo es UNA idea que el trabajo ha dejado clara, con alguna cifra solo si apoya esa idea;
  - conectores entre párrafos («Por otro lado», «Finalmente»).
- **Aportaciones:** no en una lista, sino dentro de la idea donde encajan (el conjunto etiquetado, en
  la del clasificador).
- **Limitaciones:** sin sección.
  - El único etiquetador va al capítulo 3 (Datos), en el párrafo del etiquetado.
  - Que el quién se mide en un solo torneo ya se ha dicho bastante; no se repite.
- **Trabajo futuro, como Miriam:** pocas líneas, cada una con qué se propone y para qué, encadenadas
  con «En primer lugar… En segundo lugar… Por último…».

## DECISIÓN FINAL DE JORGE SOBRE LA 6.1 (18 oct): cuatro párrafos, y así se queda

Los párrafos son: el objetivo y el sistema; el sonido; el clasificador con la pose y la pelota; y
«Finalmente», BST como punto de partida.

**Fuera, por decisión firme:**
- «Funciona lo bastante bien»: es un juicio subjetivo. Si un 75 % sirve lo decide quien use la
  herramienta (lección de Ferrovial).
- «El objetivo se cumple»: no hace falta; Miriam y sus compañeros no lo dicen.
- Que el visor lo hace usable: es evidente.
- El etiquetado: no es una conclusión.
- Derecha frente a revés, la regla del saque y la cadena: ya están en los resultados.

**Línea 11:** se acepta la versión ajustada (no depende del aspecto de la pista y funciona en pistas
no vistas). Matiz para la defensa: el clasificador no ve la imagen, pero la pose y la pelota sí, así
que un cambio fuerte de pista o de luz podría afectarle a través de ellas. La pelota se entrenó en
una sola pista.

**Después:** Jorge aplica la tabla de redacción y el capítulo 6 queda terminado. El siguiente es
Datos.

## Revisión del primer borrador de la 6.1 (18 oct): contenido acordado entonces (superado)

**Cinco párrafos:**
1. **El objetivo y el sistema que lo hace posible.** No hace falta repetir qué entrega el sistema
   (instante, jugador y tipo): todo el capítulo lo dice.
2. **El sonido.** Se quita su última frase («al ser el primer módulo…»), que repetía la idea de la
   cadena. Sin atribuir a Decorte: eso va en la revisión de autoría del capítulo 4, no aquí.
3. **El clasificador.**
   - Añadir que parte de BST (bádminton) y ha habido que adaptarlo al pádel: es trabajo de Jorge.
   - Con la pose y la pelota; en pistas no vistas.
   - El etiquetado lo hizo posible (sin decir que está disponible: el repositorio es privado).
   - Derecha frente a revés es lo difícil; la regla del saque.
4. **La cadena, en un párrafo propio.** Con su porqué: el jugador equivocado le da al clasificador el
   esqueleto de otro.
5. **«Finalmente»: se ha conseguido una herramienta utilizable.**
   - El sistema funciona lo bastante bien con una cámara fija y elevada al fondo de la pista, como la
     de las retransmisiones, y procesa al ritmo del vídeo.
   - El visor lo convierte en una herramienta usable: por jugador, golpes por tipo, mapa de calor y
     línea de tiempo con salto al golpe.
   - Así se cumple el objetivo, y el trabajo futuro lo amplía.
   - Sin el 75 %.

**Después:** la revisión de redacción y, al terminar el capítulo, la de conectores.

## 6.1 Conclusiones (propuesta inicial: cinco ideas, un párrafo cada una)

1. **El objetivo y por qué importa:** que clubes, equipos e incluso profesionales dispongan de una
   herramienta para analizar el juego a partir de una cámara fija.
   - El tipo de golpe es la pieza que faltaba: sin ella no hay estadísticas por jugador útiles para
     el análisis.
   - Además, Novillo et al. lo dejaban como trabajo futuro.
2. **El sonido es la señal adecuada para el cuándo:** la imagen no lo resolvía, y el audio detecta
   casi todos los golpes con un coste mínimo (menos del 1 % del procesado). Lección: elegir la señal
   que mejor refleja el evento.
3. **El tipo de golpe se puede aprender con solo la pose y la pelota, y funciona en pistas no vistas**
   (un torneo fuera cada vez).
   - El conjunto propio de 2377 golpes etiquetados es lo que lo ha hecho posible.
   - Lo difícil es derecha frente a revés.
   - La regla del saque muestra que el conocimiento del deporte complementa al modelo.
4. **En la cadena, la asignación es el eslabón que más pesa:** un error de jugador arrastra al tipo,
   porque el clasificador recibe el esqueleto de otro. Conclusión sobre dónde mejorar, sin repetir
   cifras.
5. **Valoración global (para cerrar, como el «Finalmente» de Miriam):**
   - funciona lo bastante bien para usarse en clubes o para analizar puntos;
   - el visor lo hace usable sin conocimientos técnicos;
   - procesa al ritmo del vídeo (unas 1,05 veces su duración).

**Opcional:** un párrafo personal (lo aprendido, construir sobre un método publicado, técnicas que
no se dominaban). Decide Jorge.

## 6.2 Trabajo futuro: versión de Jorge (18 oct)

- **Introducción:** «una serie de mejoras para ampliar las funcionalidades del sistema y su
  robustez». Prosa con «En primer lugar… En segundo lugar… En tercer lugar… Por último…», sin
  lista; cada línea con qué y para qué.
1. **Más tipos de golpe, NO solo los subtipos del remate.**
   - Prioridad: el **globo** y la **volea** (derecha o revés cerca de la red), los más frecuentes. Los
     subtipos del remate (bandeja, víbora) no son prioridad.
   - Hoy están agrupados por gesto: las voleas van como derecha o revés, y el globo también. Separarlos
     es subdividir: reetiquetar y reentrenar, sin rehacer el sistema.
   - Por qué es viable: el modelo ya recibe la pelota, y eso es lo que los distingue (la trayectoria
     alta del globo; la volea se golpea sin bote). Es una razón, no algo medido.
   - Opcional: cambiar el ejemplo de la 4.2.5 («bandeja y víbora») por el globo y la volea, para que
     coincida.
2. **Límites según la cámara:** altura, posición, ángulo y fps. PADELVIC sirve para probarlo (cuatro
   alturas, dos a ras de pista). Para qué: saber en qué instalaciones de club funciona.
3. **Partidos completos:** separar los puntos y mantener las identidades frente a los cambios de
   campo, de perspectiva y las repeticiones. Sin marcador (Jorge: no aporta).
4. **Desenlace del punto:** quién lo gana (golpe ganador) o lo pierde (error), y el golpe que lo
   prepara («asistencias»; pulir la expresión en la revisión de redacción).
- **Aviso:** la lista de tipos de golpe que pegó en `conclusiones.tex` (líneas 28-42) parece copiada
  de una web. Quitarla antes de compilar y no usarla tal cual. Si hace falta describir los golpes,
  citar una fuente académica (Ramón-Llin et al., 2020, en `notas_citas.md`).

## 6.2 Trabajo futuro (propuesta inicial: cuatro líneas)

1. **Más clases:** bandeja, víbora y dejada, subdividiendo las actuales sin rehacer el sistema (ya
   anunciado en la 4.2.5).
2. **Generalizar a otras grabaciones:**
   - otras alturas y posiciones de cámara (PADELVIC, cuatro alturas, con captura de movimiento);
   - otros fps: hoy todo está medido a 25 fps. La asignación (±4 fotogramas) y la pelota (3
     fotogramas) dependen de ellos; el clasificador no, porque su ventana va en segundos. Pasar los
     parámetros a segundos y aprovechar más fps con datos etiquetados a esa frecuencia.
3. **Partidos completos y marcador:** hoy se trabaja con puntos ya cortados. Segmentar los puntos y
   distinguir el juego real de las repeticiones.
4. **Análisis táctico sobre las estadísticas:** patrones antes de un error, en qué golpe se pierden
   más puntos; que el sistema saque conclusiones de un partido por sí mismo (anunciado en la 4.2.5).

**Descartadas para no alargar** (se pueden sumar si Jorge quiere): el procesado en directo; más
etiquetadores y el jugador anotado en más torneos.

## Para otros capítulos

- **Capítulo 3 (Datos):** el único etiquetador como limitación, en el párrafo del etiquetado
  (decisión de Jorge, 18 oct). No hay medida del acuerdo entre anotadores.

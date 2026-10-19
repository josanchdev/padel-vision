# Notas para el capítulo 2 (Estado del arte)

Referencia de trabajo mientras se escribe el capítulo; se borra al terminar la memoria.
No es texto de la memoria: es la estructura acordada y qué dice cada fuente, sacado de las notas
comprobadas por Jorge (`Downloads/papers_tfg/notas_citas.md`, con página y sección).

## Estado

- **19 oct:** estructura acordada, con cuatro secciones.
- **`pages/estado-del-arte.tex`:** ya tiene las cuatro secciones con sus etiquetas, todavía sin texto.
- **Forma de trabajar:** sección a sección, como siempre. Al final, la pasada de conectores.

## Criterio

- **Decorte et al. se cita UNA vez en el capítulo,** en la 2.1, como sistema completo de pádel. En la
  2.2 no hace falta (Jorge no quiere citarlo «8 mil veces»).
- **El hilo del capítulo:** de lo general al hueco.
  - Qué se extrae ya del pádel con vídeo.
  - Cómo se detecta el instante de un evento.
  - Cómo se reconoce una acción a partir del esqueleto.
  - Qué hay del tipo de golpe en pádel, y que nadie lo hace con vídeo publicando su método.
- **Las secciones se justifican por lo que aportan al lector,** no porque la normativa lo recomiende
  (Jorge, 19 oct).
- **El paper de los tutores (Novillo et al.) sirve de mapa,** no de texto: ni una frase suya. El
  informe de plagio lo detectaría y, además, son los tutores.
- **El informe de la FIP** (crecimiento del pádel) va en la introducción, no aquí.

## VERIFICACIÓN EN LOS PDF Y BÚSQUEDA DE MÁS TRABAJOS (19 oct)

**Criterio de Jorge:** todo se comprueba de nuevo en el PDF (las notas pueden estar anticuadas), y solo
se citan artículos de revistas o congresos reconocidos (nada de TFG, TFM, tesis, repositorios ni
revistas dudosas).

**Lo verificado (leído en los PDF de `Downloads/papers_tfg`):**
- **Javadiha et al. (2021), *Sensors* 21(10):3368.** Correcto:
  - la cámara de retransmisión, un estándar de hecho, unos 7,6 m de alto, para reducir lo que tapa
    la malla (resumen);
  - comparan detección, segmentación y pose en vídeos amateur y profesionales;
  - lo mejor es la pose *top-down*: 99,8 % de detección, error por debajo de 5 y 12 cm, como el de los
    anotadores humanos;
  - transformación de 4 puntos a partir de las esquinas de la pista, marcadas a mano, exacta solo a
    ras de suelo;
  - discusión (§4): los sensores o marcadores obligan a equiparlos y no sirven para vídeos ya
    grabados.
- **Decorte et al. (2024), CVPRW.** Correcto:
  - detección del golpe por audio (SED-net adaptada), asignación al jugador (voto ponderado en
    500 ms), reidentificación por pose y distancias entre jugadores y a la red;
  - conjunto abierto: 99 puntos de 11 torneos, resúmenes cedidos por Play Sports, con cámara fija;
  - además: la pista a mano (descartan la detección por color), YOLO con pose más una **máscara de la
    pista**, y TrackNet entrenado en tenis.
- **Novillo et al. (2025), IDEAL 2024, LNCS 15346.** Correcto:
  - pista automática (K-means por color, contorno y homografía de 4 puntos), YOLOv8 filtrando por la
    región de la pista (**centro de la caja dentro de la pista**) y TrackNet;
  - probado en PADELVIC (vídeo amateur, tres vistas; falla a ras de pista), con evaluación solo
    cualitativa, por figuras;
  - trabajo futuro, §4 pág. 119: «incorporating information about the types of shots made by
    players». También mencionan aiball y PadelCourt (pág. 111).

**Trabajos reconocidos que NO teníamos:**
1. **Javadiha, Andujar y Lacasa (2023), *Sensors* 23(1):441,** «A Query Language for Exploratory
   Analysis of Video-Based Tracking Data in Padel Matches».
   - Posiciones de los jugadores automáticas desde el vídeo; los golpes, con su tipo según una
     taxonomía de golpes, anotados de forma SEMIAUTOMÁTICA.
   - Proponen un lenguaje de consultas para el análisis táctico.
   - → 2.1 (analizar el pádel a partir del vídeo) y apoyo en la 2.4 (incluso ahí el tipo no es
     automático). Falta el PDF para comprobarlo.
2. **PadelTracker100 (Bada-Nerín et al., 2026, *Data in Brief*).** ¡Ojo! **Anota el tipo de golpe**
   en vídeo:
   - seis clases (derecha, revés, remate, saque, dejada y otro), fotograma a fotograma, en las dos
     finales del WPT de 2022;
   - es un artículo de datos: no propone ningún clasificador («Related research article: None»).
   - → 2.4. Además, **cambia lo que se puede afirmar:** NO se puede decir que no exista ningún
     conjunto de vídeo con el tipo de golpe.
3. **Papaspyropoulos et al. (2025), IEEE ISVLSI,** «Real-time Padel Strokes Classification» (DOI
   10.1109/ISVLSI65124.2025.11130340).
   - Con sensores (Arduino Nano 33 BLE), transformadas wavelet y STFT, dos CNN y un *random forest*;
     en torno al 97 % según el resumen de IEEE.
   - → 2.4, junto a Cartes Domínguez (sensores). Falta el PDF para comprobarlo.
- **Opcionales:**
  - PADELVIC (Javadiha et al., 2024, *Padel Scientific Journal*), un conjunto multicámara con captura
    de movimiento, sin tipo de golpe;
  - un capítulo de libro de 2024 del mismo grupo que Cartes Domínguez (sensores).

  Ninguno de los dos aporta nada nuevo.

**Descartados por el criterio** (solo por si pregunta el tribunal):
- IJIRMPS 2025, «Padel Analytics using Deep Learning»: trabajo de estudiantes, sin conjunto de datos
  ni evaluación;
- una tesis de máster de Oporto (2023);
- repositorios: Padex, YOLOv8-Pose más BiLSTM, el repositorio de Cartes y «padelpose» de Hugging Face.

**DECISIONES DE JORGE sobre lo encontrado (19 oct):**
- **PadelTracker100 NO se menciona en el estado del arte.** Sus etiquetas de tipo son malas (Jorge las
  probó), igual que las de Roboflow y sitios parecidos. Solo hay que cuidar que el hueco no diga que
  no existe ninguna etiqueta. Se sigue citando en los capítulos 4 y 5 para la pelota.
- **Sensores: solo Cartes Domínguez,** el de la pulsera.
- **Papaspyropoulos (2025) NO se añade.**
  - Es de acceso cerrado. El resumen solo dice que usan un Arduino Nano 33 BLE con sus sensores y
    redes neuronales, y un piloto con voluntarios, con un 97 %.
  - No dice cuántas clases ni cómo se evalúa, y no se puede comprobar sin acceso al IEEE.
  - Es un simposio de hardware (VLSI). No aporta nada sobre Cartes Domínguez.
- **Javadiha et al. (2023): SOLO en la 2.1, no en la 2.4.** Lo que hacen:
  - **Datos:** un partido, la final femenina del Estrella Damm Open del WPT de 2020 en Madrid, con
    vídeo público.
  - **Posiciones automáticas,** con su método de 2021.
  - **Los golpes, el jugador que golpea, el tipo, el resultado y si es globo,** «anotados de forma
    semiautomática», sin explicar cómo: en la práctica, una persona sobre las posiciones ya
    calculadas.
  - **Su aportación:** un lenguaje de consultas en Python para explorar tácticas, evaluado traduciendo
    10 afirmaciones de estrategia de la literatura a consultas, sin estudio con usuarios.
  - **En la 2.1:** del seguimiento al análisis táctico; las posiciones son automáticas y los golpes
    se anotan, no se detectan.
  - En la 2.4 no se repite «a mano»: allí ya está Ramón-Llin, con los expertos.

- **CORRECCIÓN (Jorge, 19 oct): Javadiha (2023) NO se menciona.** Un lenguaje de consultas para la
  táctica es el estado del arte de otra cosa, fuera del alcance.

**ESTRUCTURA FINAL (el resumen inicial de la 2.1 queda VERIFICADO en los PDF):**
- **2.1:** Javadiha (2021), Decorte (2024) y Novillo (2025).
- **La tabla de la 2.1** (lo que extrae cada uno, verificado en los PDF):

  | | Pista | Jugadores | Pelota | Instante del golpe | Jugador que golpea | Tipo de golpe |
  |---|---|---|---|---|---|---|
  | Javadiha (2021) | a mano (esquinas) | sí | no | no | no | no |
  | Decorte (2024) | a mano | sí | sí (TrackNet de tenis) | sí (audio) | sí | no |
  | Novillo (2025) | automática (color) | sí | sí (TrackNet) | no | no | no |
- **2.2:** E2E-Spot y SED.
- **2.3:** ST-GCN, PoseConv3D, TemPose y BST.
- **2.4:** Ramón-Llin (a mano, por expertos), Cartes Domínguez (sensor) y las aplicaciones
  comerciales, que llevan al hueco.

**El hueco, formulado con precisión** (que no lo contradiga nada de lo encontrado): no se ha
encontrado ningún trabajo publicado que clasifique automáticamente el tipo de golpe en pádel a partir
de vídeo.
- Los sensores sí lo clasifican (Cartes Domínguez, Papaspyropoulos).
- PadelTracker100 da etiquetas, pero no un método.
- Javadiha (2023) lo anota de forma semiautomática.
- Las aplicaciones no publican ni su método ni su precisión.

## 2.1 Análisis del pádel con vídeo

**Idea:** qué sistemas existen para analizar el pádel a partir de vídeo y qué extrae cada uno. El
hilo es que todos se quedan antes del tipo de golpe.

- **Javadiha et al. (2021)** (`javadiha2021positions`, *Sensors*):
  - estiman la posición de los jugadores desde la cámara alta de retransmisión, que en pádel es un
    estándar de hecho (unos 7,6 m de altura);
  - la pose «top-down» es lo mejor: detecta el 99,8 % de los jugadores, con un error parecido al de
    los anotadores humanos;
  - la homografía es exacta solo a ras de suelo, y los jugadores del fondo salen peor;
  - frente a los sensores, el vídeo no obliga a equipar al jugador y sirve para vídeos ya grabados
    (§4).
- **Novillo et al. (2025)** (`novillo2025padel`, los tutores):
  - extraen de un vídeo monocular la pista (automática, por color y homografía), los jugadores
    (YOLOv8) y la pelota (TrackNet);
  - proponen como trabajo futuro «incorporating information about the types of shots» (§4, pág.
    119): es lo que hace este TFG.
- **Decorte et al. (2024)** (`decorte2024padel`), la ÚNICA cita del capítulo: detectan el golpe con el
  audio, lo asignan al jugador y analizan las posiciones. No clasifican el tipo, y tampoco lo ponen
  como trabajo futuro (§7).
- **PadelTracker100:** NO va aquí; es un conjunto de datos y ya está citado en el capítulo 4.

**CAMBIO (Jorge, 19 oct): FUERA el párrafo del «escenario».** Presentaba la cámara de retransmisión
como lo que hace posible el análisis, y eso sonaba a requisito de su sistema, que no está medido
(Jorge cree que funciona con otras alturas, ángulos y posiciones; los límites son su trabajo futuro).
- El dato de la cámara (alta, unos 7,6 m, al fondo) va dentro de Javadiha, como la cámara con la que
  trabajaron ELLOS.
- La 2.1 queda en dos partes: los tres trabajos y el cierre.
- **Para la revisión final del capítulo 4:** en la 4.1, «Esta deberá estar situada al fondo de la pista
  y con una cierta elevación» suena a requisito. Quizá suavizarlo («en las grabaciones utilizadas, la
  cámara está…»). Lo decide Jorge.

### Detalle de la 2.1 (19 oct): tres partes (la primera, superada)

1. **El escenario** (un párrafo corto):
   - la cámara de retransmisión es un estándar de hecho: fija, al fondo y alta, a unos 7,6 m, para que
     la malla tape lo menos posible (Javadiha, págs. 1-2);
   - eso hace posible el análisis por vídeo;
   - dificultades: reflejos en el cristal, malla y estructura que tapan, y jugadores que se tapan
     entre sí.
2. **Los tres trabajos, por orden cronológico** (Novillo el último, para terminar en su trabajo
   futuro):
   - **Javadiha (2021):**
     - posiciones de los jugadores comparando métodos;
     - la pose, lo más fiable (99,8 %, un error como el de un anotador humano);
     - homografía de las esquinas, exacta a ras de suelo, con más error en el lado lejano;
     - frente a los sensores, el vídeo no equipa al jugador y sirve para vídeos ya grabados.
   - **Decorte (2024)**, la única cita:
     - instante del golpe por audio, asignación al jugador y análisis de posiciones;
     - publican el conjunto de datos de partida: remitir a la `\cref{sec:conjunto-partida}` en vez de
       citar otra vez.
   - **Novillo (2025):**
     - pista automática por color con su homografía, jugadores con YOLOv8 y pelota con TrackNet;
     - el tipo de golpe como trabajo futuro (§4, pág. 119).
3. **El cierre** (1-2 frases): extraen jugadores y pelota, y Decorte incluso el cuándo y el quién;
   ninguno clasifica el tipo de golpe, que es lo que proponía Novillo y aborda este trabajo. Sin
   «nosotros frente a ellos».

**Cuidado:**
- ni una frase de Novillo;
- sin las cifras de sus resultados, salvo, si acaso, el 99,8 % de Javadiha.

**Tabla:** antes o después del cierre. Antes de hacerla, comprobar en los PDF:
- **Javadiha:** ¿la pista a mano? ¿Hay pelota?
- **Decorte:** según las notas, la pista a mano, jugadores con YOLO y máscara, pelota con TrackNet de
  tenis, instante por audio y jugador que golpea.

## 2.1: ESTADO (19 oct)

Escrita con el borrador del hilo («El primer paso…», «Sobre esa base…», «Un paso más allá…», «Sin
embargo…»). Jorge la reescribirá con sus palabras al final. Sin tabla.

## 2.2: VERIFICADO EN EL PDF (19 oct) y borrador

- **E2E-Spot** (Hong et al., ECCV 2022):
  - define la detección del fotograma exacto de un evento (el contacto con la pelota en tenis, un
    bote, el despegue o el aterrizaje en patinaje), con una tolerancia de pocos fotogramas;
  - el modelo es compacto y aprende de extremo a extremo sobre las imágenes;
  - **suplementario D.1-D.2:** en tenis evalúan los eventos «por su definición temporal», no por
    atributos como derecha o revés («we separate temporally precise spotting from fine-grained
    classification»);
  - **en FineGym:** la clasificación «can be performed after events have been spotted».
- **El sonido:** SED-net (`adavanne2017sed`, ya citada y comprobada en el capítulo 4).
- **Decorte:** solo se nombra, sin cita, porque se cita en la 2.1.
- **Descartado:** Baughman et al. (2019), audio en tenis; no se pudo confirmar dónde se publicó.
- **Borrador:** un párrafo con el problema general, la vía de la imagen (E2E-Spot y la separación entre
  detectar y clasificar), la vía del sonido (SED) y el enlace con el pádel (Decorte, sin cita).

## RESUMEN DEL CAPÍTULO PARA REDACTARLO CON TUS PALABRAS (19 oct, todo verificado en los PDF)

**El hilo:** de lo general al hueco. Qué se extrae ya del pádel con vídeo, cómo se detecta el momento
de un evento, cómo se reconoce una acción con el esqueleto y qué hay del tipo de golpe en pádel.
Termina en: no hay ningún trabajo publicado que clasifique el tipo de golpe en pádel con vídeo.

**Introducción del capítulo: NO va** (Jorge, 19 oct). Es el único capítulo sin ella; los capítulos 3,
4 y 5 sí la tienen.

**Estilo de citas en ESPAÑOL** (Jorge, 19 oct; para la pasada final):
- **Presente:** «Yan et al.~\cite{} proponen ST-GCN, el primer modelo que…»;
- **sin pasiva con «por»:** es un calco del inglés;
- **«presentar», no «introducir»:** «introducir» por «presentar» es un anglicismo, según la Fundéu;
- **alternativa impersonal:** «En~\cite{} se propone…»;
- **variar el verbo** (proponen, presentan, plantean, desarrollan, aplican, amplían, adaptan) y **no
  empezar todas las frases por el autor** (mezclar con «Por su parte, X~\cite{} sustituye…»);
- **«et al.» en cursiva,** según la RAE.

**2.1 Análisis del pádel con vídeo** (dos párrafos; cada trabajo añade una capa):
- **Entrada:** varios trabajos analizan el pádel con vídeo, cada uno extrae una parte distinta.
- **Las posiciones** (`javadiha2021positions`, solo con la cita): con los puntos del esqueleto y una
  homografía, la posición en la pista sale con un error como el de un anotador humano. Comparan la
  caja, la silueta y el esqueleto, y gana el esqueleto.
- **La pista y la pelota** (Novillo et al., `novillo2025padel`): la pista automática (por color), los
  jugadores con YOLOv8 y la pelota con TrackNet, que permiten representar el juego en una pista en
  2D.
- **Cuándo y quién** (Decorte et al., `decorte2024padel`, la ÚNICA cita del capítulo): cuándo se
  golpea, por el sonido, y quién golpea, por la cercanía de la pelota.
- **El cierre:** ninguno identifica el tipo de golpe, que Novillo et al. proponían como trabajo futuro
  y en el que se centra esta memoria.
- **No decir:** que el sistema propio necesite la cámara de retransmisión.

**2.2 Detección del instante de un evento** (un párrafo, general, sin pádel):
- **El problema:** el momento exacto de un evento (un golpe, un bote, un salto) es un problema propio,
  la detección precisa de eventos.
- **Por la imagen:** E2E-Spot (`hong2022e2espot`) aprende a localizar el fotograma del evento en el
  vídeo (tenis, patinaje), con una tolerancia de pocos fotogramas. Separa localizar de clasificar: en
  tenis detecta el golpe sin distinguir derecha de revés y deja la clasificación para después.
- **Por el sonido:** la detección de eventos sonoros localiza los sonidos en una grabación
  (`adavanne2017sed`).
- **Sin Decorte y sin el sistema propio.**

**2.3 Reconocimiento de acciones con esqueleto** (dos párrafos):
- **De los píxeles al esqueleto:** se estiman los puntos del cuerpo y se clasifica su movimiento; sin
  el fondo.
- **ST-GCN** (`yan2018stgcn`): el primero con grafos; las articulaciones son nodos, unidos por el cuerpo
  y por el tiempo.
- **PoseConv3D** (`duan2022poseconv3d`): volumen de mapas de calor y CNN 3D; más robusto ante el ruido
  de la pose.
- **Por qué no basta en la raqueta:** son modelos para acciones cotidianas muy distintas, y los golpes
  son rápidos, breves y se diferencian en detalles.
- **TemPose** (`ibh2023tempose`): *transformer* para bádminton, evaluado en dos conjuntos de bádminton
  y en NTU RGB+D (no en tenis); mejora claramente al añadir la posición del volante.
- **BST** (`chang2026bst`): pose, trayectoria del volante y posición en la pista; supera a los
  anteriores en bádminton (ShuttleSet, BadmintonDB) y tenis (TenniSet); la trayectoria de la pelota
  es «una dirección prometedora». Es la base del clasificador propio (`\cref{sec:tipo}`, opcional).
- **Sin el texto de BST; sin la historia de ST-GCN y PoseConv3D en este TFG.**

**2.4 Tipo de golpe en pádel** (dos párrafos):
- **El tipo de golpe sirve para analizar el rendimiento:** Ramón-Llin et al.
  (`ramonllin2020stroke`) comparan el tipo, la dirección y la zona de 8441 golpes (1055 puntos, 9
  partidos) entre las parejas ganadoras y perdedoras, registrados a mano por dos observadores expertos
  y formados (kappa > 0,80) con un programa de vídeo (LINCE).
- **Con sensores:** Cartes Domínguez et al. (`cartesdominguez2023padel`) clasifican 13 tipos con un
  sensor inercial en la muñeca, lo que obliga a equipar a cada jugador. Es la única limitación que
  se menciona, porque es lo que lo distingue del vídeo.
  - **FUERA del texto** (Jorge, 19 oct): los ejercicios dirigidos y que no haya zurdos.
  - **Para la defensa:** se validó en ejercicios dirigidos (un tipo por prueba), no en partidos, así
    que no se sabe cómo funciona en competición. Por eso no es comparable.
- **Las aplicaciones comerciales:** aiball (cámaras en la pista, «type of hit»), Padelytics (cámara del
  club, «qué golpes usas más») y Padel AI (móvil, puntúa seis técnicas). Sin método ni precisión
  publicados. Revisadas el 19 oct; `urldate` actualizado en el `.bib`.
- **El hueco:** hasta donde se ha podido comprobar, no se ha encontrado ningún trabajo publicado que
  clasifique automáticamente el tipo de golpe en pádel a partir de vídeo.
- **No mencionar:** PadelTracker100 (etiquetas malas), Javadiha (2023) ni Papaspyropoulos (2025).

**Estilo para la pasada final** (Jorge): nombrar a los autores, como en el estado del arte de BST,
pero con el formato ESPAÑOL de arriba, en todo el capítulo, más la pasada de conectores.

## 2.2: DECISIÓN (Jorge, 19 oct)

Se quita la última frase («En el pádel… es la vía que siguen Decorte et al.»): el párrafo termina en
las dos vías.
- En el estado del arte no se nombra a Decorte ni se dice lo que hace el sistema propio.
- En el capítulo 4, la revisión de autoría tiene que seguir diciendo que el detector reimplementa el
  de Decorte (categoría 1).

## 2.3: VERIFICADO EN EL PDF (19 oct) y borrador

- **ST-GCN:** «the first to apply graph CNNs to the task of skeleton based action recognition»; aristas
  espaciales (las del cuerpo) y temporales (la misma articulación en fotogramas seguidos).
- **PoseConv3D:** volumen de mapas de calor 3D en lugar de un grafo y una CNN 3D; más robusto ante el
  ruido de la pose y generaliza mejor a otros conjuntos.
- **TemPose:** *transformer* para movimientos finos en bádminton, con capas temporales y de
  interacción; mejora claramente al añadir la posición del volante.
- **BST:** pose, trayectoria del volante y posición en la pista (a partir de las líneas); recorta los
  fotogramas de cada golpe; supera a los anteriores en ShuttleSet, BadmintonDB y TenniSet; «ball
  trajectory is a promising direction».
- **Hilo:**
  - párrafo 1: de los píxeles al esqueleto, con ST-GCN y PoseConv3D;
  - párrafo 2: en la raqueta todos los golpes son rápidos y se diferencian en detalles, así que hacen
    falta TemPose y BST.
- **Sin el texto de BST** (sus ideas sí, su texto no, ni traducido).
- **El estilo «X et al. propusieron…»,** como en BST, en la pasada final de todo el capítulo (Jorge,
  19 oct).
- **Recomendación:** no mencionar que se probaron ST-GCN y PoseConv3D; es historia del desarrollo.

## 2.2 Detección del instante de un evento (plan inicial)

**Idea:** las dos formas conocidas de detectar CUÁNDO ocurre un evento, en cualquier deporte. Uno o dos
párrafos.

- **Por vídeo, con E2E-Spot** (Hong et al., 2022; `hong2022e2espot`, ECCV):
  - define la detección precisa de eventos: el fotograma del evento, con una tolerancia de unos pocos
    fotogramas;
  - CNN 2D con desplazamiento temporal más una GRU bidireccional, entrenada de extremo a extremo
    sobre las imágenes;
  - **la clave para esta memoria:** separan a propósito localizar el evento de clasificarlo
    (suplementario D.1 y D.4). Es el mismo diseño en dos pasos de este sistema (detectar el golpe y
    después clasificarlo), y le da precedente.
- **Por sonido, con la detección de eventos sonoros (SED):** SED-net (Adavanne et al.,
  `adavanne2017sed`), una CRNN que marca cuándo ocurre cada sonido; es de la que parte el detector de
  audio (ya citado en el capítulo 4).
- **No repetir** la comparación de imagen frente a audio con la medición propia: ya está en el
  capítulo 4 (`tab:deteccion`).

## 2.3 Reconocimiento de acciones con esqueleto

**Idea:** cómo se reconoce una acción a partir de la pose, de lo general a los deportes de raqueta,
hasta BST, la base del clasificador.

- **ST-GCN** (Yan et al., 2018; `yan2018stgcn`):
  - el primer modelo con convoluciones sobre grafos: las articulaciones son nodos y las conexiones
    del cuerpo y entre fotogramas, aristas;
  - es la referencia clásica con la que se comparan TemPose y BST.
- **PoseConv3D** (Duan et al., 2022; `duan2022poseconv3d`):
  - el esqueleto como volumen de mapas de calor, clasificado con una CNN 3D;
  - más robusto que las GCN al ruido de la pose.
- **TemPose** (Ibh et al., 2023; `ibh2023tempose`):
  - un *transformer* temporal sobre el esqueleto para bádminton (token de clase, capas pre-norm);
  - la pelota entra por su propio bloque TCN;
  - BST parte de aquí.
- **BST** (Chang, 2026; `chang2026bst`):
  - pose más trayectoria de la pelota para clasificar el golpe en bádminton; la pelota es
    imprescindible;
  - ventana adaptativa;
  - base del clasificador de este trabajo (ya explicado en la 4.2.5; aquí, solo situarlo).
- **Opcional, para el contexto:** en este trabajo se probaron ST-GCN y PoseConv3D solo con la pose, y
  se pasó a pose más pelota (BST). Va en la bitácora; decidir si aporta o sobra.

## 2.4 Tipo de golpe en pádel

**Idea:** qué existe para el tipo de golpe en pádel, y el hueco. Hasta donde se ha podido comprobar,
nadie lo clasifica a partir del vídeo publicando su método.

- **Ramón-Llin et al. (2020)** (`ramonllin2020stroke`): en ciencias del deporte, el análisis de golpes
  se hace a mano.
  - 8441 golpes de 9 partidos, en 11 tipos, registrados uno a uno por dos observadores expertos con
    software de vídeo (LINCE), con un acuerdo kappa > 0,80;
  - es costoso, y eso motiva automatizarlo.
  - En las notas estaba previsto para la introducción: decidir si va aquí, allí o en los dos sitios.
- **Cartes Domínguez et al. (2023)** (`cartesdominguez2023padel`):
  - clasifican 13 tipos de golpe con un sensor inercial en la muñeca, en ejercicios dirigidos y con
    jugadores diestros;
  - **sin sus resultados:** no son comparables (sensor frente a vídeo, ejercicios frente a
    partidos, 13 clases frente a 4);
  - frase acordada en las notas: «en la literatura revisada por pares solo hemos encontrado la
    clasificación con un sensor en la muñeca [n]». PadelTracker100 también lo cita como la única
    excepción en pádel.
- **Las aplicaciones comerciales** (aiball, Padelytics, Padel AI):
  - anuncian estadísticas por tipo de golpe, pero no publican ni su método ni su precisión;
  - aiball lo cita Novillo (§1, pág. 111);
  - van en una frase, citando sus webs con la fecha de consulta (5 oct 2026; entradas `@misc` ya en
    el `.bib`).
- **No citar:** Padex (código abierto, reglas sobre la pose, sin evaluación); solo por si pregunta el
  tribunal.

## Decisiones y propuestas (19 oct)

- **Ramón-Llin:** es ciencias del deporte, sin aprendizaje automático. Dos observadores expertos
  registran a mano 8441 golpes con LINCE para estudiar qué golpes se usan según el resultado del
  punto y el lado de la pista.
  - Aporta dos cosas: que el tipo de golpe importa para el análisis, y que hoy se hace a mano y es
    costoso.
  - Propuesta: la PRIMERA frase de la 2.4 («tradicionalmente, el tipo de golpe se registra a mano…»),
    y no repetirlo en la introducción. Pendiente de que Jorge lo confirme.
- **Tablas candidatas, por script** (sin fila de «este trabajo»: la comparación propia va en
  Experimentos):
  1. **2.1, qué extrae cada sistema de pádel** (la más recomendada). Columnas: pista, jugadores,
     pelota, instante del golpe, jugador que golpea y tipo de golpe; la columna del tipo, vacía.
     Comprobar en `notas_citas.md` qué extrae exactamente cada trabajo (Javadiha, ¿solo las
     posiciones?).
  2. **2.4, cómo se ha abordado el tipo de golpe** (opcional). Columnas: trabajo, entrada, si es
     automático, si publica su método y si se evalúa en partidos.
- **Figuras:** ninguna. Las de otros trabajos no aportan y habría que reproducirlas.

## Cómo redactar la 2.1 (19 oct)

- **Frase de entrada** (Jorge eligió la 2, con algún cambio): «En los últimos años se han publicado
  varios trabajos que analizan el pádel a partir de vídeo, cada uno centrado en extraer una parte
  distinta de la información del juego».
- **Sin crónica de autores y fechas:** el sujeto de cada frase es lo que se consigue, y el autor va en
  la cita. Sin años.
- **Orden por lo que extraen, de menos a más:**
  1. las posiciones (Javadiha, solo con la cita);
  2. además, la pista y la pelota (Novillo et al., por su nombre);
  3. además, cuándo se golpea y quién (Decorte et al., por su nombre, una sola vez).

  Después, el cierre: ninguno da el tipo de golpe, que Novillo proponía como trabajo futuro.
- **Dos párrafos medianos:**
  - las posiciones y el seguimiento (Javadiha y Novillo);
  - el golpe (Decorte) y el cierre.
- **Conectores de progresión** («Además de…», «Un paso más allá…»), no «En primer lugar».
- **Para la introducción (1.1, «Motivación y antecedentes»):** la frase completa. Este trabajo parte
  del de Novillo et al. (que extraen del vídeo la pista, los jugadores y la pelota) y aborda lo que
  proponían como trabajo futuro, el tipo de golpe.
- **En el cierre de la 2.1, solo el enlace:** «…el tipo de golpe, que Novillo et al. proponían como
  trabajo futuro y que aborda este trabajo».

## Por decidir

- Ramón-Llin: ¿en la 2.4, en la introducción o en los dos sitios?
- ¿Se menciona que se probaron ST-GCN y PoseConv3D antes de BST?
- Las fuentes de la introducción de Novillo (García-Giménez 2022, Martín-Miguel 2023 y Priego
  Quesada 2013) aún no están en el `.bib`: comprobarlas antes de usarlas.

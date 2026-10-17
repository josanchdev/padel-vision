# Notas para la sección 4.2.5 (Tipo de golpe)

Referencia de trabajo mientras se escribe la sección; se borra al cerrarla.
No es texto de la memoria: es cómo funciona el clasificador, qué es de BST y qué es nuestro,
y qué falta. Todo comprobado contra el código (`shot_type_dataset.py`, `shot_type_model.py`,
`shot_type_train.py`, `rally_analysis.py`), las evidencias de `docs/metrics/` y el artículo de BST
con su material suplementario (Chang, CVPRW 2026).

## Estructura de la sección

1. Introducción: para qué sirve y las cuatro clases. **Hecho.**
2. Ventana: qué intervalo se mira, con `figuras/ventana.pdf`. **Hecho.**
3. Entrada: qué secuencias recibe el modelo y cómo se preparan, con `figuras/entrada.pdf`. **Hecho.**
4. Procesamiento: la arquitectura (BST-0). **Pendiente**, con una figura propia al estilo de la
   figura 2 de BST, generada por código.
5. Salida: softmax, regla del saque y por qué no hay pesos de clase. **Pendiente.**

Sin bloque de entrenamiento (decidido): el origen de los datos de entrenamiento ya está en la entrada,
los pesos de clase van en la salida, y los hiperparámetros y el protocolo (un torneo fuera cada vez)
van en Experimentos o en el anexo de reproducibilidad.

## Cómo funciona

### Ventana (de BST; artículo §3.1 y suplementario C.2)

- Empieza en el golpe anterior y acaba en el siguiente más ε = 0,25 s.
- Primer o último golpe del punto: ese extremo a 0,5 s del golpe (t = 0,5 s, ε = t/2).
- Tope: ningún extremo a más de 1,5 s del golpe.
- 100 pasos: se toma el fotograma más cercano a cada uno de 100 instantes equiespaciados
  (repite o descarta fotogramas según cuántos tenga la ventana). Los 100 pasos son de BST;
  cómo se llega a ellos lo explicamos nosotros (BST no lo dice).
- La ventana está en segundos, así que no depende de los fps del vídeo. El resto del sistema sí
  tiene parámetros en fotogramas (ver «Límites» abajo).

### Entrada

- Dos secuencias de 100 pasos: pose del jugador que golpea (17 puntos × x, y, confianza) y pelota
  (x, y, indicador de presencia).
- Pose, en cada fotograma: origen en el punto medio de las caderas, dividida por la longitud del
  tronco (caderas → hombros) de ese fotograma.
- Pelota: misma escala y origen, pero con la posición y la escala **medias** del jugador en la
  ventana (una referencia fija que conserva la forma de la trayectoria).
- Pelota ausente: (0, 0, 0). Antes, el posprocesado rellena huecos de hasta 6 fotogramas, que cuentan
  como presentes. Falta en el 5,4 % de los pasos de media; el 76 % de las ventanas no tiene huecos.
- Jugador ausente: se mantiene el último esqueleto. Solo al principio de la ventana, si aún no ha
  aparecido, el paso queda a cero (con confianza 0): en 38 de 2.186 golpes (1,7 %).
- Si el jugador aparece en menos de 25 de los 100 pasos, el golpe queda sin clasificar.
- **Los rivales no entran**: sus golpes solo fijan los límites de la ventana; su influencia llega
  por la pelota.
- Entrenamiento con las salidas del propio sistema (poses de YOLO, pelota propia, jugador elegido por
  la asignación, que acierta el 87,5 %); del etiquetado solo vienen el tipo y el instante. En
  funcionamiento el instante lo da el audio.

### Procesamiento (BST-0; 1.778.104 parámetros)

Hiperparámetros de BST: d_model 100, d_head 128, 6 cabezas, dropout 0,3, kernel del TCN 5.

1. Un TCN por secuencia (convoluciones temporales con dilataciones 1 y 3, BatchNorm, GELU; no cambian
   la longitud): pose 51 → 100 → 100 canales; pelota 3 → 50 → 100.
2. Encoder temporal: 2 capas de transformer (pre-norm), los mismos pesos para las dos secuencias, con
   un token de clase delante y codificación posicional sinusoidal fija. Salen un resumen de cada
   secuencia y las secuencias codificadas.
3. Atención cruzada (la idea central de BST): Q de la pose, K y V de la pelota. La pose «pregunta» a
   la trayectoria.
4. Encoder de interacción: 1 capa, con su propio token, sobre la secuencia fusionada.
5. Cabeza: concatena los tres resúmenes (pose, pelota, fusión; 300 valores) → LayerNorm → MLP →
   4 logits.

### Salida

- Softmax de 4 clases; gana la más probable.
- **Regla del saque** (nuestra): solo el primer golpe del punto puede ser saque. Los 97 saques
  etiquetados son el primer golpe de su punto. Precisión del saque 0,861 → 1,000, recall igual
  (0,916); macro-F1 0,842 → 0,859; acierto 82,48 % → 82,71 %.
  (Ojo: el docstring de `best_class` dice 0,722; es de un modelo anterior. La cifra buena es la de
  `docs/metrics/shot_type_classifier.json`.)
- **Sin pesos de clase**: el saque está 8,4:1 en desventaja, pero con la regla los pesos no ayudan:
  macro-F1 0,859 sin pesos, 0,855 con potencia 0,5, 0,857 con frecuencia inversa.
- **Sin umbral de confianza** (ADR-0016, decisiones D y E sin usar): todo golpe con jugador
  identificado recibe uno de los cuatro tipos.

### Entrenamiento (para Experimentos o el anexo, no para 4.2.5)

AdamW, lr 5·10⁻⁴, weight decay 10⁻², label smoothing 0,1, 8 épocas de calentamiento y coseno,
60 épocas, lotes de 64. Se guarda la última época (sin elegir el mejor punto). El modelo entregado
se entrena con todos los datos. Evaluación: un torneo fuera cada vez (11 rondas).

### Resultados (para Experimentos)

| | Precisión | Recall | F1 | Golpes |
|---|---|---|---|---|
| Derecha | 0,806 | 0,804 | 0,805 | 799 |
| Revés | 0,789 | 0,803 | 0,796 | 731 |
| Remate | 0,882 | 0,877 | 0,879 | 561 |
| Saque | 1,000 | 0,916 | 0,956 | 95 |

Acierto 82,71 % (IC 95 %: 80,3–85,0), macro-F1 0,859. Por torneo, de 75,4 % (Ámsterdam) a 91,7 %
(Malmö). Cadena completa: tipo 83,5 %; 89,3 % cuando el jugador está bien asignado. Confusión
principal: derecha ↔ revés. Fuentes: `shot_type_classifier.json`, `shot_type_per_tournament.json`,
`chain_cv.json`, `confidence_intervals.json`.

## La red explicada paso a paso (para entenderla, no es texto de la memoria)

### De dónde viene

- **Transformer** (Vaswani et al., 2017): el mecanismo de atención, la atención multicabeza y la
  codificación posicional sinusoidal.
- **TemPose** (Ibh et al., CVPRW 2023): transformer que clasifica golpes de bádminton a partir de la
  pose. De ahí vienen la forma de las capas (normalización antes, MLP) y la idea de un codificador
  temporal seguido de uno de interacción.
- **BST** (Chang, CVPRW 2026): parte de TemPose y añade la trayectoria de la volante como entrada
  propia, un TCN por entrada y la atención cruzada pose → trayectoria. BST-0 es su versión base; las
  grandes añaden PPF, Clean Gate y Aim Player, que no usamos.
- **Nuestro modelo**: BST-0 con un solo jugador, pelota con indicador, codificación posicional fija
  y 4 clases.

Notación (la misma en la figura `clasificador.tex`, el texto y las ecuaciones): T = 100 pasos;
J = 17 puntos clave; D = 100 (cada paso es un vector de 100 números dentro de la red); C = 4 clases;
H = 6 cabezas; D_A = 128 (dimensión de cada cabeza). La figura usa T, J, D y C; H y D_A solo
aparecen en las ecuaciones.

### Formas de los datos

| Paso | Pose | Pelota |
|---|---|---|
| Entrada | 100 × 51 (17 puntos × x, y, confianza) | 100 × 3 (x, y, indicador) |
| Tras el TCN | 100 × 100 | 100 × 100 |
| Con el token de clase | 101 × 100 | 101 × 100 |
| Tras el codificador temporal | resumen (100) + secuencia (100 × 100) | igual |
| Tras la atención cruzada | secuencia fusionada (100 × 100) | |
| Tras el codificador de interacción | resumen de la fusión (100) | |
| Cabeza | 300 → 4 | |

### 1. TCN (red convolucional temporal)

- **Qué es**: una convolución como las de una CNN de imágenes, pero en una dimensión: el filtro se
  desliza a lo largo del tiempo en vez de por el plano de la imagen.
- **Cómo**: cada filtro mira 5 pasos (kernel 5) de todos los canales de entrada, multiplica por sus
  pesos y suma; da un número por paso. 100 filtros → 100 rasgos por paso.
  y_t^(c) = b^(c) + Σ_c' Σ_{k=−2..2} W_k^(c,c') · x_{t+d·k}^(c'), con d la dilatación.
- **Dilatación**: la 1.ª capa mira 5 pasos seguidos (d = 1); la 2.ª, 5 pasos de 3 en 3 (d = 3), y
  abarca 13 con los mismos pesos. Juntas, cada paso de salida depende de 17 pasos (campo receptivo):
  17/99 de la ventana, 0,34 s en una de 2 s y 0,52 s en la máxima de 3 s.
- **Capas**: pose 51 → 100 → 100; pelota 3 → 50 → 100. Tras cada convolución, BatchNorm (normaliza
  cada canal: entrenamiento estable), GELU (no linealidad, una ReLU suave) y dropout del 30 %.
  El relleno mantiene los 100 pasos.
- **Por qué**: pasar de coordenadas crudas a rasgos de movimiento local antes de la atención, y
  llevar las dos entradas (51 y 3 valores) al mismo tamaño D = 100. En BST sustituye al módulo
  inicial de TemPose.
- **Parámetros**: 76.100 (pose) y 26.200 (pelota).

### 2. Token de clase y codificación posicional

- **Token de clase**: 100 números aprendidos que se ponen delante de la secuencia (101 pasos). No
  viene de los datos; en el codificador mira a todos los pasos y a la salida su vector es el resumen
  de la secuencia. En nuestro código, el mismo token para pose y pelota. 100 parámetros.
- **Codificación posicional**: la atención trata la secuencia como un conjunto (desordenar los pasos
  no cambiaría nada). Para darle el orden, a cada paso se le suma un vector que solo depende de su
  posición t:
  PE(t, 2i) = sin(t / 10000^(2i/D)),  PE(t, 2i+1) = cos(t / 10000^(2i/D)).
  Cada par de dimensiones oscila a una frecuencia distinta: cada posición tiene una firma única y las
  cercanas se parecen. Es la del Transformer original, fija (0 parámetros); BST la aprende. Cambio
  sin motivo documentado ni efecto medido. Se suma en tres sitios: antes del codificador temporal
  (101), de la atención cruzada (100) y del de interacción (101). Tras la primera, dropout del 30 %.

### 3. Codificador temporal (Trans 1)

- **La atención, intuitiva**: cada paso hace una pregunta (Q), ofrece una etiqueta (K) y un
  contenido (V). Se compara la pregunta de un paso con las etiquetas de todos (producto escalar), los
  parecidos se convierten en pesos que suman 1 (softmax), y el nuevo vector del paso es la media de
  los contenidos con esos pesos. Así el impacto se trae información de la preparación aunque esté
  50 pasos antes.
- **Fórmulas** (una cabeza): Q = X·W_Q, K = X·W_K, V = X·W_V (X: 101 × 100; cada W: 100 × 128).
  A = softmax(Q·Kᵀ / √D_A) es 101 × 101 (fila i: cuánto mira el paso i a cada paso; suma 1).
  Salida A·V. Dividir por √D_A evita que los productos crezcan con la dimensión y saturen el softmax.
- **Multicabeza**: 6 veces en paralelo con pesos distintos (cada cabeza puede fijarse en algo
  distinto); se concatenan (6 × 128 = 768) y W_O lo devuelve a 100:
  MHSA(X) = [SA_1 ‖ … ‖ SA_6]·W_O.
- **La capa** (pre-norm): X̃ = X + MHSA(LN(X)) y X' = X̃ + FFN(LN(X̃)).
  LN normaliza cada vector (media 0, varianza 1, con dos parámetros por dimensión). El «X +» es la
  conexión residual: la capa aprende una corrección sobre lo que había, lo que facilita apilar
  capas. La FFN es una red pequeña aplicada a cada paso por separado (100 → 400 → 100, GELU): la
  atención mezcla pasos; la FFN transforma cada uno.
- **Dos capas** (L = 2), y **los mismos pesos para pose y pelota** (se pasan por separado): las dos
  quedan en el mismo «idioma» para la atención cruzada, con la mitad de parámetros. BST hace igual.
- **Salida**: el vector del token (resumen) y los 100 de los pasos (secuencia).
- **Detalles**: W_Q, W_K, W_V sin sesgo; dropout del 30 % en los pesos de atención y tras W_O.
- **Parámetros**: 388.200 por capa (atención 307.300, FFN 80.500, normalizaciones 400) → 776.400.

### 4. Atención cruzada (CrossTrans)

- La misma fórmula, pero las preguntas salen de la pose y las etiquetas y los contenidos de la
  pelota: Q = LN(P)·W_Q, K = LN(B)·W_K, V = LN(B)·W_V.
  A = softmax(Q·Kᵀ / √D_A) es 100 × 100: fila t, qué instantes de la pelota importan al instante t
  de la pose. Z = [softmax(Q_h·K_hᵀ / √D_A)·V_h]_h ·W_O, y la salida es Z + FFN(LN(Z)).
- **Sin residual con la pose**: la salida está hecha solo de contenidos de la pelota, elegidos y
  pesados por el gesto. Es «la pelota vista por el jugador».
- **Por qué**: la idea central de BST. La pose sola es ambigua (gestos parecidos en el esqueleto
  mandan la pelota a sitios distintos) y la trayectoria lo resuelve; la atención cruzada deja que el
  gesto elija qué momentos de la trayectoria importan. En su tabla 1, añadir la volante es lo que
  más sube (0,776 → 0,810 de acierto en TemPose, más que la posición en pista).
- **Una capa** (L = 1) y **una rama** (BST: una por jugador). Antes, codificación posicional a las
  dos secuencias. 388.400 parámetros.

### 5. Codificador de interacción (Trans 2)

Un token propio (otro vector aprendido), codificación posicional y una capa de transformer igual que
las del paso 3, con pesos propios. Resume la secuencia fusionada en un vector. 388.200 parámetros.

### 6. Cabeza

Concatena los tres resúmenes, pose, pelota y fusión (3 × 100 = 300); LN, y MLP 300 → 400 (GELU,
dropout) → 4. Esos 4 números (logits) son puntuaciones sin escala; el softmax de la salida los
convierte en probabilidades. Se usan los tres porque el tipo depende del gesto, de la trayectoria y
de su relación. 122.604 parámetros.

### Detalles generales

- **Dropout del 30 %**: en el TCN, tras sumar la posición antes del codificador temporal, en la
  atención y la FFN, y en la cabeza. Solo en entrenamiento; en uso se apaga (y BatchNorm usa las
  medias aprendidas).
- **Inicialización**: capas lineales Xavier uniforme con sesgo 0; convoluciones Xavier normal;
  tokens con ruido normal de desviación 0,02.
- **Hiperparámetros** (D 100, D_A 128, H 6, L 2 + 1 + 1, dropout 0,3, kernel 5, FFN × 4): los de
  BST (los L están en su figura 2; el resto, según nuestro código, de su implementación). No se han
  ajustado.
- **Cómo aprende** (no va en 4.2.5): entropía cruzada con suavizado 0,1 sobre los logits, AdamW.

### Tamaño

| Módulo | Parámetros |
|---|---|
| TCN de la pose | 76.100 |
| TCN de la pelota | 26.200 |
| Tokens de clase (2) | 200 |
| Codificador temporal (2 capas) | 776.400 |
| Atención cruzada | 388.400 |
| Codificador de interacción | 388.200 |
| Cabeza | 122.604 |
| **Total** | **1.778.104** |

La mayor parte está en las proyecciones de la atención (6 cabezas × 128 = 768, más que D = 100).

### Estilo de redacción: Miriam como referencia (decidido el 16 oct)

La subsección se reescribe con el estilo de la memoria de Miriam (versión anterior guardada en el
commit 59d2ede). Patrón: (1) qué determina el fenómeno, en palabras; (2) la idea antes de la
fórmula; (3) frase fija que anuncia la ecuación y acaba en dos puntos; (4) cada símbolo definido
justo después; (5) aplicación al caso concreto, ya con sus números; (6) ampliación paso a paso;
(7) interpretación. Frases largas encadenadas con conectores, no una idea por línea.

- Anunciar ecuaciones: «toma la siguiente forma:», «se obtiene la siguiente expresión:»,
  «expresándose como:», «definiéndose como:», «según la siguiente fórmula:».
- Definir símbolos: «donde X representa…, es decir, …», «donde X es el vector de…, que contiene…»,
  «El término contenido dentro de… representa…».
- Conectores: «Dado que», «Por tanto», «En consecuencia», «Por este motivo», «De esta forma»,
  «Para ello», «A partir de ella», «Esto equivale a».
- Avanzar: «Para aplicar esta lógica en el caso concreto de…», «Tomando como ejemplo…», «Este
  planteamiento se repite para…», «Una vez definido…», «Al trasladar… a…».
- Interpretar: «reflejando que», «La única excepción es», «se interpreta como».

Números: la ecuación general va sin ellos; al aplicarla al caso concreto, sí (como sus «15 ramas»,
«6 condensadores»). Sin figura en la salida (descartadas la de atención cruzada y la del punto).

### Dos versiones del procesamiento (plan de Jorge, 16 oct)

Se termina primero la versión LARGA (la que se escribe juntos, módulo a módulo, con ecuaciones);
después Jorge redacta una CORTA a su manera (un párrafo sencillo por parte y solo la ecuación
imprescindible, la de la atención). La larga no se borra: si el tutor pide más detalle, se
recupera (propuesto: commit con la versión larga antes de simplificar).

### Criterio de redacción del procesamiento (Jorge, 16 oct)

Que se entienda qué hace cada módulo; nada de valores en el texto (H, D_A, D sin número: se dicen
con palabras qué son, como hace BST; los valores al anexo de reproducibilidad). Fuera de las
ecuaciones, palabras («se repite 6 veces»), no letras. La ecuación de la capa (LN + residual) se
quita: se cuenta en dos frases. Siglas inglesas con `\ac{}` (añadidas tcn, ffn, mhsa).

### Pendiente para el texto del codificador temporal

- Tras explicar la atención: «el token, como cualquier paso, toma información de todos los demás;
  como no tiene contenido propio, a la salida contiene el resumen de la secuencia» (el *cómo* del
  token, que se quitó de su párrafo para no adelantar la atención).
- La D = 100 ya está definida en el párrafo de la TCN.

### Notación y ecuaciones (decidido el 16 oct)

Las ecuaciones mandan; la figura se adapta. Con esta elección las letras de la figura (T, J, D, C)
ya encajan.

1. TCN, con la notación estándar de Bai, Kolter y Koltun (2018, `\cite{bai2018tcn}`; comprobado en
   el PDF: sección 3.3, ec. 2, definida para una secuencia 1-D x ∈ ℝⁿ; sección 3.2: su TCN es
   causal, «1D FCN + causal convolutions». Jorge lo revisa al terminar): la suya es causal, F(s) = Σ_{i=0}^{k−1} f(i)·x_{s−d·i}; la nuestra mira a los
   dos lados, así que va centrada: F(s) = Σ_{i=0}^{k−1} f(i)·x_{s+d(i−⌊k/2⌋)}, k = 5 (posiciones
   s−2d … s+2d). La suma sobre los canales de entrada se dice en el texto.
2. Codificación posicional: PE(t, 2i) = sin(t / 10000^(2i/D)), PE(t, 2i+1) = cos(…).
3. Atención: Atención(Q, K, V) = softmax(Q·Kᵀ / √D_A)·V, con Q = X·W_Q, K = X·W_K, V = X·W_V;
   MHSA(X) = [cabeza_1 ‖ … ‖ cabeza_H]·W_O (el ‖ de la leyenda de la figura).
4. Capa: X̃ = X + MHSA(LN(X)), X' = X̃ + FFN(LN(X̃)) (BST, ec. 5-6).
5. Atención cruzada: Q = LN(P)·W_Q, K = LN(B)·W_K, V = LN(B)·W_V; Z como en la 3; salida
   Z + FFN(LN(Z)). P = secuencia de la pose, B = de la pelota (de *ball*); se explica en el texto
   o, si no, en el pie de la figura.
6. Salida (bloque 5): p_c = e^{z_c} / Σ_{c'=1}^{C} e^{z_{c'}}.

Símbolos: T pasos (100), J puntos clave (17), D dimensión del modelo (100), D_A dimensión de cada
cabeza (128), H cabezas (6), C clases (4), d dilatación (1 y 3); índices t (paso), k
(desplazamiento), i (dimensión), h (cabeza), c (clase).

### Qué fórmulas irían en la memoria

1. La capa de transformer (pre-norm), citando a BST y TemPose.
2. La atención multicabeza, citando a Vaswani y BST.
3. La atención cruzada, de BST, en nuestra forma de una rama.
4. La codificación posicional sinusoidal, de Vaswani (es el cambio).

La del TCN es opcional: basta con decir kernel 5, dilataciones 1 y 3, y 17 pasos.

## Qué es de BST y qué es nuestro

| | BST (bádminton) | Este trabajo (pádel) |
|---|---|---|
| Ventana: límites, ε, tope, extremos, 100 pasos | ✔ | Se adopta |
| Jugadores en la entrada | Los dos (sus clases incluyen quién golpea: ShuttleSet duplica cada tipo en arriba/abajo) | Solo quien golpea; ya lo decide la asignación, y en pádel hay cuatro |
| Atención cruzada | Una rama por jugador | Una sola rama |
| Pelota | Posición 2D | Relativa al jugador + indicador de presencia |
| Pose | Articulaciones y huesos (su tabla 1; sin comprobar para BST-0) | Articulaciones + confianza |
| Clases | 25/35 de bádminton | 4 de pádel + regla del saque |
| Datos | ShuttleSet, anotaciones dadas | 2.377 golpes etiquetados en este trabajo; entrada generada por el propio sistema |
| Codificación posicional | Aprendida | Sinusoidal fija |

Las decisiones que importan para la defensa son las de jugador único, pelota relativa con
indicador, clases y regla del saque, y entrenar con las salidas del sistema. Las demás son detalles
que no hace falta contar.

## Comprobaciones hechas (no versionadas)

- **Escala de la pose**: dentro de un golpe varían parecido el tronco (8,7 %), cadera-tobillo (9,6 %)
  y la altura de la caja (8,1 %); los tobillos se detectan con la misma confianza (0,98 frente a
  0,99). El tronco es una convención razonable, no una ventaja medida; no decir que es mejor.
  Si se quiere justificar: «no incluye las extremidades que ejecutan el golpe».
- **fps**: los 99 puntos están a 25 fps. Asignación (±4 fotogramas) y pelota (3 fotogramas
  seguidos, huecos y filtros en fotogramas) dependen de los fps; el clasificador no.
  → Conclusiones: limitación (todo medido a 25 fps; otro vídeo habría que convertirlo antes) y
  trabajo futuro (parámetros en segundos; aprovechar más fps con datos etiquetados a esa frecuencia).

## Figuras

- `figuras/ventana.pdf`: esquema al estilo de la figura 1 (c)+(d) de BST; `scripts/build_window_figure.py`.
- `figuras/entrada.pdf`: lo que recibe el modelo en un remate de MALMO_02 (fotograma 615), estela de
  ±3 fotogramas; `scripts/build_input_figure.py`. Para el pie: el hueco entre la mano y la pelota es
  la pala, que el modelo no recibe.
- Pendiente: figura de la arquitectura (bloque 4).

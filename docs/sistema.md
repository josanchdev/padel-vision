# Padel Vision — cómo funciona el sistema

Ficha técnica del sistema completo: qué hace cada paso, de dónde sale, qué se
midió y cuánto tarda.

---

## El problema

Dado el vídeo de un partido de pádel grabado desde la cámara fija habitual de
retransmisión, responder automáticamente tres preguntas por cada golpe:

1. **¿CUÁNDO?** — el instante exacto del golpe
2. **¿QUIÉN?** — cuál de los cuatro jugadores lo ejecutó
3. **¿QUÉ TIPO?** — derecha, revés, remate o saque

Las dos primeras las resuelve el trabajo de referencia (Decorte et al., CVPRW
2024). **La tercera es la aportación propia**: ese paper se detiene en la
detección binaria y no clasifica el gesto.

---

## El flujo en 5 pasos

```
Vídeo (mp4 con audio)
   │
   ├─1─ AUDIO ──────────► CRNN ──────────► instantes de golpe        F1 0,956
   │
   ├─2─ IMAGEN ─────────► YOLO26-pose ───► esqueletos + identidad J1-J4
   │                      + máscara de pista + re-identificación
   │
   ├─3─ IMAGEN ─────────► TrackNetV3 ────► trayectoria de la pelota  F1 0,94
   │                      + limpieza física (parábola local)
   │
   ├─4─ 1+2+3 ──────────► voto ponderado ► QUIÉN golpeó       87,46% jugador
   │                                                          92,16% equipo
   │
   └─5─ pose + pelota ──► BST-0 ─────────► TIPO de golpe      81,84% acc
                          + regla del saque                   0,847 macro-F1
```

### Paso 1 — CUÁNDO: detección por audio

**Para qué sirve.** Es el disparador de todo lo demás: marca los instantes que
merece la pena analizar. Sin él habría que examinar los 90.000 frames de un
partido; con él, solo los ~1.500 en los que ocurre algo.

**Qué hace.** Convierte el audio a espectrograma log-Mel (40 bandas) y una red
convolucional-recurrente (CRNN) marca, frame a frame, si suena un golpe.

**Por qué audio y no imagen.** Un golpe de pala produce un chasquido muy
característico. Detectar el instante exacto por imagen es de los problemas
abiertos de la visión por computador: en tenis, el estado del arte (E2E-Spot)
acierta el frame exacto solo el 45% de las veces. El sonido lo resuelve casi
gratis.

**De dónde sale.** Reimplementado del paper de Decorte et al. (arquitectura
SED-net adaptada). **Se probó primero el camino propio** —detectar el golpe por
pose y trayectoria de pelota— y se descartó midiéndolo: F1 0,821 frente a 0,956
del audio sobre los mismos datos.

**Resultado:** F1 **0,956** (el paper reporta 0,92).

### Paso 2 — Los jugadores

**Para qué sirve.** Es el paso que alimenta a los dos últimos: el paso 4 necesita
saber dónde están las muñecas de cada jugador para medir su distancia a la
pelota, y el paso 5 recibe directamente la secuencia de esqueletos del golpeador
como entrada del clasificador. Convierte "hay píxeles de gente" en "éstos son los
cuatro jugadores, éste es J3, y así se está moviendo".

Hace tres cosas:

**a) Extraer los esqueletos.** YOLO26-pose localiza a las personas y devuelve 17
articulaciones de cada una (muñecas, codos, hombros, caderas, tobillos…).

**b) Descartar al público.** El detector encuentra *personas*, y en un partido
hay cientos: grada, árbitro, cámaras. Una máscara construida con la homografía de
la pista conserva solo a quien pisa dentro; sin ella el sistema podría atribuir
un golpe a alguien sentado en las gradas.

**c) Saber quién es quién.** Esto es lo menos evidente y lo más importante. El
detector no sabe que son siempre los mismos cuatro: en cada frame encuentra
"cuatro personas" sin memoria de las anteriores. Si un jugador queda tapado un
instante, al reaparecer sería alguien nuevo. Sin resolver esto no se puede
afirmar "J3 ha jugado 12 derechas", porque no habría forma de saber que esas 12
son de la misma persona. El componente de identidad mantiene los números J1-J4
estables durante todo el partido y los recupera tras las oclusiones.

**Lo propio aquí.** La numeración inicial y la re-identificación se replican del
paper (por posiciones de aparición y desaparición, no por apariencia: los
compañeros visten igual). Se añadió el *puente de huecos*: el detector pierde a
un jugador uno o dos frames, y el frame del golpe tiene la misma probabilidad que
cualquier otro de ser uno de ellos. Medido sobre el ground truth, el golpeador
real faltaba de su propio frame de golpe en el 10% de los casos.

### Paso 3 — La pelota

**Para qué sirve.** La pelota es la prueba de quién golpeó: en el instante del
golpe está pegada a la pala de alguien. El paso 4 decide por proximidad, y el
paso 5 la usa como segunda entrada — es lo que distingue un remate de una
defensa alta, que en el esqueleto se parecen pero mandan la pelota en
direcciones opuestas. Es la entrada que más aporta al clasificador según las
mediciones publicadas de BST.

**Qué hace.** TrackNetV3 localiza la pelota frame a frame; después se limpia la
trayectoria con un modelo físico (entre golpe y bote la pelota es un proyectil,
así que su recorrido en imagen es casi parabólico).

**Lo propio aquí.** Tres cosas:
- El detector está **entrenado sobre pádel** (F1 0,94). El paper usa un TrackNet
  preentrenado en tenis.
- **Resolución de inferencia a 768×432** en vez de 512×288. La red es totalmente
  convolucional, así que admite un frame mayor sin reentrenar: las detecciones
  "congeladas" (el detector enganchado a un objeto estático) bajan del 14,3% al
  8,9%. **Este cambio, de una línea, dio +6,9 puntos de asignación.**
- Limpieza física de la trayectoria: rechaza detecciones que no siguen la
  parábola, rellena huecos y suaviza. La continuidad de la estela mejora 3,4×.

### Paso 4 — QUIÉN golpeó

**Para qué sirve.** Sin esto, el sistema sabría que hubo un golpe pero no de
quién, y ninguna estadística por jugador sería posible. También decide de qué
jugador se recorta el esqueleto que recibirá el clasificador.

**Qué hace.** En una ventana de ±4 frames (160 ms) alrededor del golpe, mide la
distancia de la pelota a las muñecas de cada jugador y decide por voto ponderado (los frames
donde la pelota está más cerca pesan más). Votar sobre varios frames en vez de
uno solo lo hace robusto a que falte la pose o la pelota en el instante exacto.

**Lo propio aquí.** Dos cosas sobre el método del paper:

- Medir la distancia en **alturas de cuerpo** en vez de píxeles. Un jugador del
  fondo se dibuja pequeño, así que los mismos píxeles significan mucha más
  distancia real para él; sin normalizar, durante un remate —con la pelota
  alta— el voto se lo llevaba sistemáticamente quien estaba al fondo.
- **Ventana de ±4 frames en vez de los 500 ms del paper.** Lejos del impacto la
  pelota no está cerca de nadie en particular y el voto añade ruido: medido
  sobre su propio ground truth, estrechar la ventana da +1,9 puntos. Se probó
  también desplazarla y hacerla asimétrica —por si hubiera desfase entre audio e
  imagen— y ambas salieron peores.

**Resultado:** **87,46%** por jugador, **92,16%** por equipo (paper: 83,70% y
86,83%), sobre su mismo ground truth de 319 golpes anotados.

### Paso 5 — QUÉ TIPO (la aportación propia)

**Para qué sirve.** Es lo que convierte "hubo un golpe de J3" en información
útil para un entrenador: cuántas derechas juega cada uno, cuántos remates
resuelve, qué lado se le busca al rival.

**Qué hace.** Una red BST-0 (dos redes temporales convolucionales, un
transformer y cross-attention entre pose y pelota) clasifica el gesto en
derecha, revés, remate o saque. Recibe una ventana adaptativa que va del golpe
anterior del rival al siguiente, de modo que ve el gesto completo —preparación,
impacto y terminación— y no un frame suelto.

**De dónde sale.** Arquitectura reimplementada de BST (Chang, CVPRW 2026) sobre
bloques de TemPose, adaptada a pádel: un solo jugador en vez de dos (el golpe ya
está atribuido) y un indicador de presencia en la pelota.

**Lo propio aquí.**
- **El dataset**: 2.377 golpes etiquetados a mano, uno a uno. El dataset público
  trae el instante de cada golpe pero no su tipo.
- **Sin pesos de clase**, contra la práctica habitual. El saque está 8,4:1 en
  desventaja y la intuición dice compensarlo; medido, ponderar empeoraba incluso
  al propio saque (macro-F1 0,786 con pesos frente a 0,812 sin ellos).
- **La regla del saque**: solo el primer golpe de un peloteo puede ser un saque.
  Verificado sobre las etiquetas, los 97 saques lo son sin excepción. Es una
  regla del reglamento, no algo que el modelo deba adivinar: su precisión pasa
  de 0,722 a 1,000 sin perder ni un saque real.

**Resultado:** accuracy **81,84%**, macro-F1 **0,847**, en validación cruzada
dejando torneos enteros fuera.

---

## Modelos: qué es de quién

| Modelo | Función | Origen | Entrenado por nosotros |
|---|---|---|---|
| CRNN de audio | cuándo | arquitectura de Decorte et al. | **sí** |
| YOLO26-pose | jugadores | Ultralytics, pesos COCO | no |
| TrackNetV3 | pelota | arquitectura TrackNet | **sí**, sobre pádel |
| BST-0 | tipo de golpe | arquitectura de Chang | **sí** |

Tres de los cuatro están entrenados en este trabajo. El único de terceros es
el detector de personas, que se usa con sus pesos originales.

Además, dos componentes que no son modelos: la **asignación de golpe a jugador**
(voto ponderado) y la **homografía de pista**, que se marca a mano con seis
clics por torneo, como en el paper (ADR-0018).

---

## Resultados

Cada cifra se mide sobre los datos que la columna indica. Detección y asignación
usan el ground truth publicado con CVSPORTS_Padel y el mismo protocolo que el
paper; el tipo de golpe usa el etiquetado propio.

| Métrica | Este trabajo | Paper de referencia | Datos |
|---|---|---|---|
| Detección de golpes (F1) | **0,956** | 0,92 | CVSPORTS, 99 rallies (media de 3 ejecuciones); criterio de emparejamiento distinto, ver nota |
| Asignación — jugador | **87,46 %** | 83,70 % | CVSPORTS, VIGO, 319 golpes |
| Asignación — equipo | **92,16 %** | 86,83 % | CVSPORTS, VIGO, 319 golpes |
| Clasificación de tipo (accuracy) | **81,84 %** | *no lo hace* | etiquetado propio, 2.377 golpes, por torneos |
| Clasificación de tipo (macro-F1) | **0,847** | *no lo hace* | ídem |

Componente propio, sin cifra equivalente en el paper (no se compara):

| Componente | Resultado | Datos |
|---|---|---|
| Detección de pelota (F1) | **0,94** | PadelTracker100: entrenada con la final masculina, validada con la femenina. El paper usa un modelo de tenis sin medirlo en pádel |

Cómo leer la comparación con el paper:

- **Asignación: comparación directa.** Son los mismos 319 golpes (los 16 rallies
  de la tabla 3 del paper, con el mismo número de golpes cada uno) y la misma
  métrica: acierto medio ponderado por golpes, contando como fallo los golpes
  que no se asignan, en ambos casos. Se evalúa con los instantes **anotados**,
  no con los detectados, para medir ese paso aislado, y con la configuración
  exacta del sistema (pista marcada a mano, umbral de detección de personas
  0,25).
- **Las mejoras de la asignación se eligieron midiendo sobre esos mismos 319
  golpes** (normalizar la distancia por altura, la ventana de ±4 frames, la
  resolución del detector de pelota), así que la cifra puede ser algo optimista.
  Son pocas decisiones, y cada una tiene una explicación física, pero el sesgo
  existe.
- **Detección: mismo orden de magnitud, no una comparación exacta.** El paper
  empareja eventos por su inicio y su fin (métrica event-based de sed_eval) y
  promedia 4 particiones de los rallies; aquí se empareja por el instante del
  golpe. Con criterios distintos, 0,956 frente a 0,92 no demuestra una mejora.
- **Tipo de golpe y pelota no se comparan con el paper**: el paper no
  clasifica golpes ni mide su detector de pelota en pádel. Solo se compara donde
  hay la misma prueba.

### El sistema completo, de una vez

La tabla anterior mide cada paso aislado. Esta mide la cadena entera como
funciona en uso real: el audio decide cuándo, el voto decide quién en el
instante que encontró el audio, y el clasificador decide qué. Cada torneo se
evalúa con modelos (audio y clasificador) entrenados **sin él**.

| Métrica | Sistema completo |
|---|---|
| Detección de golpes (F1) | **0,948** |
| Jugador (VIGO) | **79,9 %** |
| Equipo (VIGO) | **89,2 %** |
| Tipo de golpe (accuracy) | **83,7 %** |
| Golpes detectados y con el tipo correcto | **80,3 %** |
| Detectado, jugador y tipo correctos (VIGO) | **68,5 %** |

Las cifras bajan respecto a los pasos aislados porque los errores se encadenan:
un golpe no detectado no puede asignarse, y un golpe asignado al jugador
equivocado le da al clasificador el esqueleto de otro. El jugador se mide en
VIGO porque es el único torneo en el que el paper anotó quién golpea (16
rallies). Evidencia: `docs/metrics/chain_cv.json` y sus figuras.

---

## Coste de procesado

Medido sobre un peloteo real (1.608 frames, 1080p, 25 fps) en una RTX 3090:

| Etapa | Tiempo | % |
|---|---|---|
| Audio (CRNN) | 1,0 s | 1,1 % |
| Pose (YOLO26n) | 52,4 s | 57,8 % |
| Pelota (TrackNetV3) | 37,3 s | 41,1 % |
| **Total** | **90,7 s** | |

**Factor: 1,41× tiempo real.**

| Duración del vídeo | Tiempo de proceso |
|---|---|
| 1 minuto | 1,4 min |
| 5 minutos | 7 min |
| Partido de 60 min | ~85 min |

El audio, que resuelve la pregunta más difícil, cuesta el 1% del total. El gasto
está en la visión, y dentro de ella la pose domina sobre la pelota.

---

## Lo que queda fuera del alcance

Declarado a propósito, no por omisión:

- **Marcador y resultado del punto**: requiere segmentar puntos y distinguir
  juego real de repeticiones, una pila de robustez documentada en el backlog.
- **Bandeja, víbora y dejada** como clases propias: se descartaron para mantener
  cuatro clases bien separadas; hay evidencia publicada de que reducir clases
  mejora sustancialmente (BST: +6,6 puntos al pasar de 35 a 25 clases).
- **Vídeos de cámara móvil o a nivel de pista**: todo el sistema asume la cámara
  fija elevada de retransmisión.

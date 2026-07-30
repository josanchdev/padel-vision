# Bitácora de experimentos

Registro cronológico de entrenamientos, problemas y decisiones experimentales.
Materia prima para los capítulos de experimentos y lecciones de la memoria.
Convención: cada entrenamiento del detector de pista es `court_vN`.

## Detección de pista (ADR-0001)

### Resumen de entrenamientos

| Run | Dataset | Resultado | Problema encontrado | Fix aplicado |
|-----|---------|-----------|--------------------|--------------|
| court_v0 | 767 frames WPT auto-etiquetados | ✅ WPT: 0,21 m error medio | Zero-shot en PADELVIC: mitad lejana desplazada, court-level alucinado | → necesidad de diversidad (batch 1 CVAT) |
| court_v1 | v0 + 40 frames PADELVIC anotados | ❌ Puntos dispersos incluso en train | **Conflicto de orientación**: labels auto usaban convención del GT (near=mitad superior), labels humanos convención cámara (near=inferior). Supervisión contradictoria | Normalización 180° de labels auto (la pista es simétrica bajo rotación); gate RANSAC 4→6 inliers |
| court_v2 | v1 + propagación cámara-fija (500+100 PADELVIC) | ⚠️ PADELVIC resuelto (4 alturas trazan bien) pero WPT empeoró 0,22→0,44 m | **Divergencia**: val_loss 1,18 (ep10) → 5,68 (ep60); box mAP colapsó a 0,03. Causa: caja envolvente degenerada (banda fina) en vistas court-level | Caja = frame completo |
| court_v3 | = v2 con caja frame completo | ❌ Estancamiento: pose mAP 0,17, early stop ep28 | **Escala OKS rota**: la loss de keypoints divide por el área de la caja; con caja gigante los gradientes se desvanecen | Caja envolvente con tamaño mínimo (35% de cada dimensión) |
| court_v4 | = v3 con caja min-size | ❌ Convergencia excelente hasta ep16 (val_pose 0,25, el mejor de la serie) pero **colapso catastrófico en ep19** (val_pose 9,8, explosión de gradientes) sin recuperación. best.pt (ep~16) da 0,78 m en WPT: sin la fase final de LR bajo no hay precisión fina | Cambio de estrategia: fine-tuning desde v1 (experto en WPT) con lr0 bajo |
| court_v5 | = v4, init desde v1 best, lr0=0.002 cos | ❌ Colapso en ep9 pese a LR 5× menor → LR descartado como causa. Pesos sin NaN → no es overflow. Cámaras sin deriva (<1 px en 73 min, medido) → labels propagados correctos | Bisección empírica |
| probe_combined | = v4 data, 1280/batch16, 18 ep | ✅ Estable (val_pose 0,11, mAP 0,99) → el detonante de las explosiones era 1920+batch8 con este dataset. PERO: error WPT 0,70 m con sesgo radial hacia el centro — el modelo predice una pista "encogida" | Diagnóstico: **prior de memorización** — los 500 frames casi-duplicados enseñan a memorizar el layout en vez de mirar la imagen; el prior contamina la precisión en WPT |

| court_v6 | WPT + 40 manuales, SIN propagación, 1280/b16 | ✅ **Modelo definitivo Nivel 1**: 54 ep estables, **0,196 m** en WPT (inferencia 1280; a 1920: 0,246). Archivado como `runs/archive/court_broadcast_v6.pt` | Cierra ADR-0006: detector para vistas arbitrarias + StaticCourtStage para cámaras fijas |

**Insight de v6:** el modelo entrenado a 1280 rinde mejor infiriendo a 1280
(0,196 m) que a 1920 (0,246 m) — al contrario que v0. La resolución de
inferencia óptima depende del entrenamiento concreto: medir siempre ambas.

**Insight de la sonda (para la memoria):** los datos propagados de cámaras
fijas son etiquetas perfectas pero ejemplos casi idénticos; en exceso,
enseñan al modelo a recitar de memoria la pista "promedio" en vez de
localizarla en la imagen. El síntoma es un sesgo radial (pista encogida) en
el dominio no-memorizable. Además, la detección con esos lotes correlacionados
es inestable a 1920+batch8 (explosiones en ep9-19) y estable a 1280+batch16.
Corolario: para cámaras FIJAS el detector es innecesario — la homografía se
calcula una vez de la anotación y es exacta para siempre; el detector solo
aporta valor en vistas nuevas/móviles.

**Insight de v4 (para la memoria):** al imponer caja mínima del 35%, el OKS
se volvió más indulgente (normaliza por área) y el pose mAP se infló a 0,95
sin que la precisión en píxeles acompañara. Las métricas estándar pueden
inflarse por decisiones de formato de datos; la métrica de tarea (error en
metros sobre la pista) es la única señal fiable de progreso.

### Hallazgos clave (para la memoria)

1. **Auto-etiquetado por homografía GT**: los 13 keypoints de pista se generan
   proyectando sus coordenadas de mundo por la H⁻¹ del GT de PadelTracker100.
   767 frames etiquetados con cero clics. Verificación visual obligatoria:
   detectó el bug de orientación.
2. **Conflicto de convenciones entre fuentes de datos** (v1): dos datasets
   pueden ser individualmente correctos y conjuntamente contradictorios. La
   métrica agregada (pose mAP 0,97) lo ocultaba porque el val estaba dominado
   por el dominio mayoritario. Lección: evaluar por dominio, no en agregado.
3. **Propagación en cámaras fijas** (v2): dispersión medida de <1 px en los
   puntos anotados entre frames → una anotación por cámara se estampa sobre
   150 frames con jugadores distintos. Convirtió 4 etiquetas útiles en 600.
   Es el modelo de despliegue real para la URJC (anotar su pista una vez).
4. **La caja del objeto "pista" es puro andamiaje** pero su escala importa:
   la loss OKS de keypoints se normaliza por el área de la caja. Caja
   degenerada → loss explota (v2); caja gigante → gradientes se desvanecen
   (v3). Ni el mAP ni la loss agregada señalaron la causa: hizo falta leer
   la trayectoria época a época y conocer la formulación de la loss.
5. **El gate de calidad de homografía necesita ≥6 inliers**: con 4, RANSAC
   acepta ajustes degenerados de puntos mal colocados (verificado
   visualmente en v1). La homografía puede ser auto-consistente y estar
   completamente equivocada; el error de reproyección interno no basta como
   única señal de calidad.

### Métricas de referencia

- Benchmark WPT (val = final femenina completa, 229 frames, GT NTT Data):
  error medio de posición en pista. v0: **0,206 m** (mediana 0,200, p95 0,248)
  con inferencia a 1920 (a 1280: 0,316 m — el sesgo de ~5 px de resolución
  vale 10 cm).
- Cadena pose+tobillos+H_GT vs posiciones GT: **0,109 m** medio (99,1%
  matching) → cota inferior alcanzable; el detector de pista añade el resto.

## Tracking e identidad

- ByteTrack de serie: jugadores del fondo (~60 px) sufren ID switches — el
  salto de un remate desplaza la caja más que su tamaño y el IoU cae a 0.
  Config propia (`padel_bytetrack.yaml`): match_thresh 0,95, buffer 120.
  Resultado: 4/4 jugadores estables en un rally completo (407 frames).
- En vídeo real (30 s, entre puntos incluidos): 16 track IDs para 4 personas.
  Solución: identidad J1-J4 anclada a la geometría (mitades de pista en
  metros), los tracks huérfanos se heredan tras 2 s. Resultado: exactamente
  4 identidades, presencia 84-92% de frames.
- Limitación documentada: los equipos cambian de lado entre juegos; la
  identidad es estable dentro de cada periodo de lado.

## Detección de golpes (dummy Nivel 1)

- Diseño: pico de velocidad de muñeca normalizada por longitud de torso
  (comparable entre jugador cercano y lejano), suavizado, máximo local con
  umbral y periodo refractario. Define el contrato `ShotEvent` que el
  clasificador de Nivel 2 rellenará.
- Evaluación sobre esqueletos GT vs 440 golpes GT (final femenina):

  | Umbral | Precisión | Recall | F1 |
  |--------|-----------|--------|-----|
  | 0,4 | 0,28 | 0,94 | 0,43 |
  | 0,6 | 0,28 | 0,83 | 0,42 |
  | 0,8 | 0,28 | 0,68 | 0,40 |

- Lectura: generador de propuestas de alto recall. El clasificador de
  Nivel 2 hereda dos misiones: tipo de golpe + rechazo de falsos candidatos
  (arquitectura en dos etapas). Estos números son el baseline formal.

## Clasificador de golpes (Nivel 2, ADR-0008)

### Atribución de golpe a jugador (Decisión 0)

Validada sobre poses GT de la final femenina (440 eventos de golpe):

- **439/440 golpes atribuidos** (100%): solo 1 run sin pelota anotada.
- Distancia muñeca-pelota en el frame de impacto: **mediana 29 px, p90 55 px**
  — la muñeca del jugador atribuido está pegada a la pelota, confirmando que
  la proximidad identifica correctamente a quien golpea (verificado también
  visualmente en derecha/revés/remate).
- Duración media de un evento: 17 frames (~0,6 s a 30 fps).
- Distribución por evento (canónica, 5 clases): Forehand 147, Backhand 115,
  Smash 114, Serve 34, Other 29 (incluye la única Dropshot). ~440/partido,
  ~880 en total → dataset modesto pero viable para baseline.

Nota: las categorías shot-event del ball.json (smash-event, etc.) están
declaradas pero VACÍAS; la única señal de golpe es el CSV por frame. De ahí
la necesidad de atribuir por pelota.

### Baseline ST-GCN (cross-match: train masculina, val femenina)

Dataset: 1346 clips (32×17×3), 6 clases, poses propias YOLO26. Split por
partido (jugadores no vistos en val) + class weights inversa-frecuencia.

- **macro-F1 0,526 · accuracy 0,633** (val = final femenina completa).
- F1 por clase: NoShot 0,75 · Smash 0,68 · Backhand 0,63 · Forehand 0,55 ·
  Serve 0,44 · **Other 0,11** (era de esperar: clase cajón-desastre + 65 clips).
- Matriz de confusión: los golpes bien separados entre sí; las confusiones
  principales son Forehand↔Backhand (esperable, gestos parecidos de perfil) y
  Smash→NoShot (el remate a veces se solapa con movimiento sin golpe).

Lectura: baseline decente para 1346 clips con test cross-jugador (honesto, no
memoriza). Los golpes "de verdad" (derecha/revés/remate) rondan 0,55-0,68 de
F1; las clases débiles son Other (mal definida) y Serve (68 clips). Techo
limitado por tamaño de dataset — el dataset PROPIO futuro lo subirá. Archivado
en `runs/archive/stgcn_shots.pt`. Siguiente: PoseConv3D para comparar.

### Ablación: data augmentation (flip L-R con swap de índices, rotación, escala)

ST-GCN + augmentation, mismo split cross-match:
- **macro-F1 0,547 (+0,021) · accuracy 0,656 (+0,023)** vs sin augmentation.
- Sube sobre todo en golpes reales: Smash 0,68→0,75, Forehand 0,55→0,59,
  Other 0,11→0,20. Serve baja (clase de 68 clips, ruidosa).
- Confirma: con dataset pequeño, augmentar da mejora real y barata. Se usa en
  toda la comparativa de arquitecturas.

### Comparativa de arquitecturas (val = final femenina, cross-jugador)

Mismo split, class weights, augmentation. **La comparación es la contribución
científica** (ADR-0008): baseline clásico vs moderno en igualdad de condiciones.

| Modelo | macro-F1 | accuracy | Fore | Back | Smash | Serve | Other | NoShot |
|--------|:--------:|:--------:|:----:|:----:|:-----:|:-----:|:-----:|:------:|
| ST-GCN (2018) | 0,526 | 0,633 | 0,55 | 0,63 | 0,68 | 0,44 | 0,11 | 0,75 |
| ST-GCN + aug | 0,547 | 0,656 | 0,59 | 0,63 | 0,75 | 0,36 | 0,20 | 0,76 |
| **PoseConv3D + aug** | **0,632** | **0,693** | 0,66 | 0,68 | 0,75 | **0,75** | 0,18 | 0,78 |

**Conclusión:** PoseConv3D bate a ST-GCN por **+8,5 pts de macro-F1**. La
mejora más llamativa es Serve (0,36→0,75): los heatmaps + CNN 3D distinguen el
saque, que ST-GCN confundía. Los golpes principales suben todos. "Other" sigue
mal (clase cajón-desastre, ~65 clips) — es problema de datos, no de modelo.
Ambos limitados por el tamaño del dataset (1346 clips); el techo lo subirá el
dataset propio. Modelo elegido para producción: **PoseConv3D**
(`runs/archive/poseconv3d_shots.pt`).

Dificultad técnica resuelta: la generación de heatmaps al vuelo con bucles
Python era demasiado lenta (timeout); se vectorizó con broadcasting numpy
(172 ms/batch), reduciendo el entrenamiento a minutos.

**Confirmación del techo de pose-solo + lección de honestidad (29 jul 2026).**
Antes de comprometer el rediseño pose+pelota (ADR-0013) medimos de nuevo el
clasificador solo-pose sobre golpes LOCALIZADOS por GT (localización perfecta,
aísla la clasificación). Reentreno limpio desde cero en ambos sentidos del split
cross-match: **val=partido 0 → macro-F1 0,62 / acc 0,67; val=partido 1 → 0,59 /
0,65.** Simétrico y ~0,60 en ambos → el techo de pose-solo es real, no depende del
split. **Lección:** al evaluar el checkpoint archivado (`poseconv3d_shots.pt`,
entrenado con val=0) sobre el partido 1 salía 0,94 — un ESPEJISMO: ese partido
estaba en su conjunto de entrenamiento, así que no era validación. Confirma por
qué el split cross-match honesto es imprescindible (ADR-0008) y por qué no basta
con accuracy. Matriz de confusión (split 0, su validación legítima): la confusión
derecha↔revés existe pero es moderada (~15 casos cada dirección); el desastre es
"Other" (F1 0,18, cajón de sastre — problema de etiquetado, no de modelo). **Este
0,60 es el baseline que el clasificador pose+pelota (Modelo 2, ADR-0013) debe
batir; la comparativa pose-solo vs pose+pelota es la contribución científica.**

Baseline honesto versionado en `runs/shot_baseline/` (val0 + val1, sin data
leakage), métricas en `baseline_metrics.json`. **F1 por clase (val0 / val1):**
Serve 0,75/0,78 · Smash 0,73/0,71 · Backhand 0,69/0,64 · **Forehand 0,60/0,59** ·
Other 0,20/0,11 · NoShot 0,77/0,73. **Lectura que refuerza la tesis pose+pelota:**
los golpes flojos son **Forehand y Backhand** — justo los que la trayectoria de
la pelota debería desambiguar (altura y lado del contacto), mientras que Smash y
Serve (gestos ya distintivos en pose) van bien. Es exactamente donde se espera
que pose+pelota aporte. (Reentreno: poseconv3d, 40 epochs, augment, cross-match.)

### Integración en el pipeline (arquitectura en dos etapas)

El clasificador sustituye al dummy vía la interfaz `ShotEvent`. Detección de
golpes en inferencia = dos etapas, **sin ninguna anotación externa**:

1. **Cuándo + quién** (señal propia): pico de velocidad de muñeca POR jugador
   (recall alto, precisión baja) propone candidatos. Que sea por jugador da la
   atribución sin pelota: quien golpea es aquel cuya muñeca aceleró.
2. **Qué** (PoseConv3D): recorta la ventana centrada de 32 frames del jugador
   propuesto y la clasifica; los falsos positivos del paso 1 caen en NoShot y
   se descartan.

Resuelve el principio de independencia de datasets: la pelota anotada de
PadelTracker100 solo etiquetó el training set (offline); en producción el
sistema es autónomo. Demo verificada: overlay "J2: Remate" etc. con tipos
reales y minimapa. Detalle: la clasificación se emite ~0,5 s tras el impacto
(necesita frames posteriores para la ventana centrada) — latencia aceptable
para análisis batch.

## Detección de pelota (Nivel 3, ADR-0009)

### Enfoque: TrackNet (heatmap temporal) sobre detección frame-a-frame

La pelota mide ~8×8 px en 1080p (medido en las cajas COCO de PadelTracker100):
un detector normal la pierde. Seguimos TrackNet: la red mira 3 frames
consecutivos y predice un **mapa de calor** de la pelota, usando el movimiento
entre frames como señal. Comparamos **TrackNetV2 (2020, baseline citable)** vs
**TrackNetV3 (V2 + refinador de oclusiones)** con el mismo protocolo — segunda
comparativa baseline-vs-moderno propia del TFG.

### Reto de datos: cajas → heatmaps (Decisión C del ADR)

Sus anotaciones son cajas (formato detección); TrackNet entrena con heatmaps
gaussianos. Se convierte el centro de cada caja en una gaussiana 2D sobre una
rejilla a ¼ de resolución. Decisión de diseño adicional: las coordenadas se
guardan **fraccionales** [0,1], no en píxeles, para que el cache de frames sea
independiente de la resolución (el tamaño de rejilla del heatmap es una
elección de entrenamiento, no queda grabada en el cache).

### Hallazgo importante: el flag `occluded` de PadelTracker100 está vacío

El plan era desglosar las métricas en pelotas visibles vs ocluidas (donde
esperábamos que V3 destacara). Al inspeccionar las anotaciones: el atributo
`occluded` existe en el esquema COCO pero **está a `False` en las 19.386
anotaciones** de la final masculina. No podemos derivar el desglose de su
etiquetado. Consecuencia para la memoria: la comparativa V2-vs-V3 se sostiene
por F1/precisión/recall global y error de localización; el desglose por
oclusión requeriría etiquetado propio (candidato para el dataset propio del
TFG, coherente con [[feedback-no-dataset-dependency]]). La infraestructura de
desglose queda lista para cuando existan esos datos.

### Infraestructura (probada end-to-end)

Cache de frames reanudable (shards de 1000, 512×288) por los crashes de WSL;
dataset `BallClips` que solo emite ventanas de frames estrictamente
consecutivos (no mezcla movimiento a través de huecos ni entre partidos);
métrica de pico-con-tolerancia (convención TrackNet); MLflow (primera vez en el
repo — backend sqlite, el file store quedó deprecado en MLflow 3) + PNGs
autocontenidos para la defensa. Extracción de 4000+3000 frames: ~66s + ~45s
(WSL estable). 86% de frames con pelota anotada.

### Integración: `BallDetectionStage` sobre vídeo real

La etapa bufferea 3 frames (TrackNet los necesita), corre el modelo, coge el
pico del heatmap y lo proyecta a metros de pista con la homografía. Vive en
`packages/ml` (lazy import) para no meter torch en `cv`. Verificado sobre metraje
WPT real: **198/200 frames con pelota detectada a 0,83-0,99 de confianza**, pelota
dibujada en overlay y minimapa junto a poses+pista+golpes. El modelo de prueba
corta ya localiza bien.

### Detección de botes: trayectoria vertical en píxeles (Decisión F)

Un bote = valle en la trayectoria vertical de la pelota (mínimo de altura =
máximo de y en píxeles). Se trabaja en píxeles de imagen, NO en metros
proyectados: la homografía mapea el plano del suelo, así que una pelota en el
aire proyecta con error creciente con la altura y distorsionaría el valle. La
homografía solo sitúa el bote confirmado en la pista.

**Hallazgo clave (composición de subsistemas):** no todo cambio de dirección es
un bote — un jugador golpeando también invierte la pelota. Se descartan los
candidatos cercanos a un golpe. Pero esto expone una dependencia: la calidad de
los botes depende de la **precisión del clasificador de golpes**. Con el dummy
(precisión 0,28, muchos falsos positivos) el guard de golpes se come TODOS los
botes (0 detectados en 200 frames). Con el clasificador PoseConv3D real (alta
precisión) los botes reales sobreviven: en el mismo tramo, el candidato en el
frame 634 (bote de suelo lejos de cualquier golpe) se detecta correctamente,
mientras los del 509/585 se descartan por coincidir con golpes reales. Es decir:
**los tres subsistemas (pelota + golpes + pista) se necesitan mutuamente**, y el
Nivel 2 no es solo un entregable previo sino un requisito del Nivel 3.

Distinguir bote de suelo vs pared/cristal queda para más adelante
(`docs/backlog.md`).

## Capa de datos estructurados (ADR-0010)

**Insight que reorienta el producto:** el vídeo procesado es *show* (verificación
visual); el valor real está en los DATOS estructurados que permiten tomar
decisiones — como en Ferrovial el entregable era el CSV de detecciones, no el
vídeo con cajas. Hasta ahora el pipeline generaba toda esa información y la
tiraba (solo la pintaba en frames). Ahora se persiste.

`MatchAnalysis` acumula por partido: posiciones de jugador (metros), golpes,
pelota (imagen + metros), botes. Serializa a **JSON canónico** (`schema_version`,
lo que consume la API/web) + **CSV** (una fila por evento, formato Excel del
entrenador). La extracción de datos se separó del renderizado de vídeo: modo
`--no-video --data-out` corre sin escribir vídeo.

Verificado sobre metraje real (200 frames): JSON con 796 posiciones de jugador,
10 golpes, 198 de pelota, 1 bote; CSV con filas tipo
`shot,509,17.5,3,Smash,0.99` (segundo 17.5, jugador 3 hizo un Smash). Un
entrenador filtra/agrupa en Excel sin ver el vídeo — esto es la plataforma, no
el reproductor. Base sobre la que se construirán las secciones de la web
(tabla filtrable, mapa de calor, timeline) y donde encajará la futura pila de
robustez como columnas más (¿cámara en pista? ¿punto en juego?).

## Detección de escena y diversidad de datos (28 jul 2026)

Al abordar la robustez ("¿cámara en pista?") se probó reutilizar la señal del
detector de pista (homografía válida = frame de juego) como detector de escena.

**Hallazgo 1 — los WPT no tienen el problema.** 200 frames repartidos por la
final masculina WPT: **100% pista**. PadelTracker100 es un feed de broadcast
casi estático (por eso NTT lo eligió para tracking); no tiene repeticiones ni
planos de público con los que validar detección de escena.

**Hallazgo 2 — con vídeo real variado, el problema aparece.** Se descargaron 3
partidos de YouTube (uso académico, `data/raw/youtube/`): un highlights
(Match of the Century) y dos full games en pistas no-WPT. La señal de pista da:
Match of the Century **5% pista** (95% gráficos/repeticiones/primeros planos),
City's Cup **20%**. El problema es real fuera del feed limpio.

**Hallazgo 3 — el detector de escena por homografía funciona PERO con sesgo.**
Inspección visual de frames: rechaza bien gráficos de título y primeros planos
(correcto), pero da **falsos negativos en pistas de color distinto al entrenado**
(la pista negra/naranja del City's Cup, vista de juego válida, se rechaza como
"no-pista"). El detector de pista (entrenado solo con azul WPT + 40 PADELVIC) no
generaliza a pistas nuevas → keypoints insuficientes.

**Matiz clave (Jorge):** un vídeo = una pista (nunca cambia a mitad). Así que el
sesgo NO es "detectar cambio de pista" (irrelevante) sino que el detector debe
poner bien los keypoints en CUALQUIER pista de un vídeo dado. Solución: más datos
de entrenamiento diversos (estos vídeos), no una arquitectura de cambio de escena.

**Prioridad de mejora (Jorge):** viendo los vídeos, donde más falla el sistema es
la **detección de pelota** (modelo actual = prueba corta, solo WPT, 6 epochs).
Los vídeos nuevos son munición kickstarter para mejorar pelota, pista, golpes
(incl. dejada) y validar escena/segmentación — respetando que son material de
entrenamiento/prueba, nunca dependencia (ADR-0005).

## Mejora de la pelota — Fase 1 (28 jul 2026)

El modelo de pelota era de prueba corta (4k frames WPT, 6 epochs, F1 0,91 en su
propia pista). Se aborda una mejora seria con estas piezas:

- **Augmentation de color** (brillo/contraste/tono, foto­métrico, igual en los 3
  frames de la ventana): generaliza a pistas/iluminaciones distintas sin
  etiquetar. Nunca mueve la pelota (el heatmap no se transforma).
- **Submuestreo de negativos** (`neg_ratio`, def. 2,0 en train): un partido
  completo es ~64% sin pelota; demasiados negativos hunden el recall (0,98, el
  punto fuerte). Val mantiene la distribución real para métricas honestas.
- **Cache memmap** — problema de ingeniería resuelto: los ~100k frames (44 GB)
  no caben en 19 GB de RAM (cargarlos causaba OOM, probable causa de crashes de
  WSL) y leer .npz comprimido era 1,9 s/ventana. `consolidate_to_memmap`
  reescribe los shards como un uint8 memory-mapped (`frames.dat`) + `meta.npz`;
  `BallClips` lee frames sueltos del disco al instante. Medido: **2,6 ms/ventana
  (era 1900), RSS 1,0 GB (iba a 44), época ~2 min**. Cross-match completo =
  99.883 ventanas.

**Punto óptimo de rendimiento (RTX 3090, medido).** Con el memmap, el cuello ya
no es la RAM sino alimentar la GPU. Benchmark de throughput (win/s, época de 54k):
batch=8/workers=4 → 73 (12,3 min); **batch=32/workers=8 → 141 (6,4 min, ~2×)**;
batch=64/workers=12 → 56 (16 min, contraproducente: batches enormes no ayudan a
un modelo pequeño y saturan la lectura). Defaults subidos a batch=32, workers=8.

**Comando del entrenamiento completo** (a lanzar vigilando el WSL):

```bash
uv run padel-ball-train --compare \
  --train-dir data/datasets/ball_cache/finalM \
  --val-dir   data/datasets/ball_cache/finalF \
  --epochs 40 --augment \
  --out runs/ball_full --plots runs/ball_plots_full \
  --mlflow-uri sqlite:///runs/mlruns.db
```

Usa los defaults óptimos (batch=32, workers=8). Entrena V2 vs V3 (cross-match:
train masculina, val femenina), con augmentation y negativos balanceados. Compara
con la prueba corta (0,91). ~40 epochs × 2 modelos × ~6,4 min ≈ 8,5 h (aceptable,
como en la vida real). Falta al cerrar Fase 1: validar visualmente sobre los
vídeos de YouTube de pistas nuevas.

**Estado (28 jul, noche):** V2 completo entrenado — `runs/ball_full/tracknetv2.pt`,
**best val_f1 0,942** (vs 0,896 de la prueba corta). Se paró tras V2 (calor/ruido
toda la noche). **Falta entrenar V3** — comando (solo V3, sin `--compare`):

```bash
uv run padel-ball-train --model tracknetv3 \
  --train-dir data/datasets/ball_cache/finalM \
  --val-dir   data/datasets/ball_cache/finalF \
  --epochs 40 --augment \
  --out runs/ball_full/tracknetv3.pt \
  --plots runs/ball_plots_full \
  --mlflow-uri sqlite:///runs/mlruns.db \
  --resume   # reanuda desde <out>.ckpt si WSL crashea (guarda cada epoch)
```

Con V3 entrenado: comparativa V2 vs V3 + Fase 1d (`padel-ball-generalization`).
**Decisión Jorge:** V3 (moderno, refinador de oclusiones) será el modelo de
PRODUCCIÓN si confirma su mejora (esperado; en prueba corta 0,911 vs 0,896),
igual que se eligió PoseConv3D sobre ST-GCN en golpes. V2 queda como baseline de
la comparativa (evidencia científica). El criterio final es el F1 con el
protocolo completo, no la asunción — se elige con el número delante.

**Por qué super-entreno con WPT si el objetivo es generalizar (razonamiento
estratégico, cuestión planteada por Jorge).** Los ~100k frames WPT NO generalizan
por sí solos a pistas nuevas (más azul no enseña negro). El super-entreno vale
por tres cosas distintas: (1) **base para fine-tuning** — el modelo aprende del
volumen "qué es una pelota, cómo se mueve, cómo separarla del fondo"; luego 1-2k
frames nuevos etiquetados bastan para especializarlo a pistas nuevas VÍA
FINE-TUNING (no desde cero: 2k frames solos no aprenderían). Es el patrón del
dataset de vehículos de noche en Ferrovial. (2) **Mejor pre-labeling** — un
modelo robusto en azul pre-anota rápido los vídeos de pistas parecidas, ahorrando
etiquetado. (3) **Métrica base honesta** — para afirmar "los datos nuevos mejoran
X" hace falta un "antes" bien medido, no un modelo cutre de prueba. Conclusión: el
volumen da conocimiento general; la diversidad (pocos datos nuevos) da
especialización. Ambos se necesitan.

**Estrategia de etiquetado (Fase 2, decidida).** Pre-label = acelerador
oportunista, NO sustituto. Por pista: si el modelo Fase 1 propone bien (>~50%),
corregir; si propone basura (<~40%, p.ej. pista muy distinta), etiquetar de cero
(más rápido y GT más limpio; evita el sesgo de anclaje). **El conjunto de TEST
se etiqueta SIEMPRE de cero a mano** (sin pre-label) para que la métrica no esté
contaminada por el propio modelo. Etiquetar es el trabajo que hace mejorar todo;
no se esquiva.

**Comparativa de generalización (Fase 1d, herramienta lista).** Sin GT en pistas
nuevas no hay F1, pero la **tasa de detección** (% frames con pelota) + la
**confianza media** cuantifican cuánto cae el modelo fuera de WPT.
`padel-ball-generalization` corre el detector sobre varias pistas y emite tabla +
JSON + gráfico de barras (para slides) + vídeos de verificación. A lanzar al
terminar el entreno:

```bash
uv run padel-ball-generalization --model runs/ball_full/tracknetv3.pt \
  --court wpt=data/raw/2022_BCN_FinalM_1.mp4 \
  --court highlights=data/raw/youtube/match_century.mp4 \
  --court pista_negra=data/raw/youtube/citys_cup.mp4 \
  --max-frames 3000 --videos -o runs/ball_gen
```

Cuantifica el hallazgo previo (WPT 100% vs pistas nuevas 5-20% con el detector de
PISTA) ahora para el detector de PELOTA, y es la métrica base honesta para medir
la mejora tras la Fase 2 (fine-tuning con datos nuevos).

## Detección de golpes por pelota + hallazgo del estado del arte (29 jul 2026)

**Detección por pelota (ADR-0012).** El pico de velocidad de muñeca daba falsos
positivos y mal timing (validado en partido real). Se sustituyó por: golpe =
cambio de dirección de la pelota + muñeca cerca. Comparativa sobre 600 frames del
partido real: muñeca 21 golpes (J1:3 J2:9 J3:1 J4:8) → pelota 15 golpes (J2:8
J4:7). **Bajaron los falsos positivos** (bueno) pero se pierden golpes reales de
J1/J3 (el umbral de proximidad muñeca-pelota es frágil, y depende de que la
pelota esté detectada justo en el frame del impacto).

**Validación métrica de la detección por pelota (29 jul 2026, `padel_ml.shot_eval`).**
Antes de rediseñar nada, medimos el detector ADR-0012 contra el GT de
PadelTracker100 (`*_shots.csv`), con GT de pose+pelota (aísla el ALGORITMO, no
nuestros detectores). El GT marca `has_shot=1` en BLOQUES de frames (~11-17f);
cada bloque = un golpe real, impacto ≈ centro. Un golpe emitido cuenta como
acierto si cae a **±3 frames** del centro (criterio estricto de timing).
Resultados (GT-based, techo teórico del método):

| Partido | Golpes reales | Recall det. | Precisión det. | F1 |
|---|---|---|---|---|
| FinalM | 466 | **15,0 %** | 16,1 % | 15,6 % |
| FinalF | 440 | **51,6 %** | 31,2 % | 38,9 % |

**El método de detección por cambio-de-dirección (ADR-0012) es insuficiente**, y
falla con GT PERFECTO de pelota → no es culpa de los detectores, es el método.
Diagnóstico de las dos causas raíz:

1. **Timing pobre.** Con tolerancia ±15f el recall de FinalM sube a 49 % (los
   eventos existen, pero caen lejos del impacto). Es exactamente la queja del
   partido real ("dice golpe antes de dar a la pelota"). Barrido de tolerancia
   FinalM: ±1f→6,7 % · ±3f→15 % · ±5f→22,7 % · ±10f→39,9 % · ±15f→49,4 %.
2. **Dependencia de trayectoria continua.** FinalM tiene 253 huecos en la pelota
   GT (max 99f sin anotar), FinalF 1138. Cada hueco rompe el cálculo de
   velocidad → giros inventados o perdidos. El método NECESITA ver la pelota
   continuamente alrededor del impacto — justo cuando la pelota va más rápida y
   se ocluye. Además el impacto físico no está en una posición fija dentro del
   bloque GT (73 % en FinalM, 43 % en FinalF), así que ni el "centro" es
   referencia estable.

**Conclusión de detección:** el cambio-de-dirección de la pelota NO es una señal
fiable para *localizar* el impacto. Sirve para reducir falsos positivos frente al
pico de muñeca, pero su recall es demasiado bajo para el TFG. La detección del
golpe hay que replantearla (ver estado del arte abajo: BST/TemPose no "detectan"
el impacto por heurística — parten de clips ya recortados en torno al contacto;
la localización del golpe es un problema separado que merece su propio enfoque).

**Hallazgo clave (Jorge, revisión del estado del arte).** El problema de fondo NO
es solo la detección: la CLASIFICACIÓN del tipo de golpe (PoseConv3D, solo pose)
falla — confunde derecha/revés/remate. Jorge cuestiona si pose es el enfoque
correcto. La literatura de deportes de raqueta lo confirma:

- **ST-GCN** (2018, solo pose): baseline, el nuestro de Nivel 2.
- **TemPose** (CVPR 2023): pose + posición + **trayectoria de pelota**, fusión
  temprana vía TCN + transformer factorizado. ~76,8%.
- **BST** (CVPR **2026**, código oficial en GitHub): pose + trayectoria de pelota
  con **cross-attention** pose↔pelota + segmentación centrada en el contacto.
  ~77,1%, estado del arte.

**Conclusión unánime de la línea (y diagnóstico de nuestro fallo):** solo pose se
satura porque "el gesto de golpear se parece entre tipos" (cita BST). **La
trayectoria de la pelota es la señal que desambigua el tipo de golpe** — todos
los métodos SotA la incorporan; "aprovechar la trayectoria de pelota es LA
tendencia en deportes de raqueta" (BST). Encaja con que PadelTracker100 anota
pose Y pelota: ambas hacen falta.

**Implicación para el TFG.** El instinto de Jorge era correcto: pose-solo no es el
enfoque bueno para el TIPO de golpe. El rumbo es **evolucionar a pose + pelota**
(replicando TemPose/BST). Ya tenemos la pelota funcionando (el trabajo de estas
sesiones era el ingrediente que faltaba) y ya la usamos para DETECTAR el golpe;
el siguiente paso es usarla también para CLASIFICAR el tipo. Contribución
científica: comparativa pose-solo (nuestro PoseConv3D) vs pose+pelota. Decisión
de arquitectura pendiente (futuro ADR): investigar más el diseño de la fusión
(TemPose fusión temprana TCN vs BST cross-attention) antes de comprometer.
Fuentes: BST arXiv 2502.21085 (CVPR 2026), TemPose CVPR 2023, comparativo de
pádel (ML/DL shot classification).

**Lectura del código oficial de BST (29 jul 2026) → arquitectura decidida
(ADR-0013).** Al leer el repo de BST se vio algo que reorienta el frente: BST y
TemPose NO detectan el golpe; parten de clips ya recortados en torno al contacto
(ShuttleSet anota el frame de golpe). El SotA **separa localizar de clasificar**.
Nosotros los mezclábamos en la heurística ADR-0012, que hace mal las dos cosas
(recall 15-52%). Insight de Jorge: un humano detecta el golpe sin esqueleto ni
reglas, viendo partidos; la arquitectura correcta es la de detección industrial
de Ferrovial — un **detector** genérico alimenta a un **clasificador**
especializado (camiones → tipo de camión). **Decisión (ADR-0013): dos modelos
aprendidos.** Modelo 1 = detector de golpe (pose+pelota → ¿golpe en este frame?,
objetivo recall ~85-95%, entrenado con los 906 bloques `has_shot` de
PadelTracker100). Modelo 2 = clasificador de tipo (pose+pelota estilo BST/TemPose
sobre el clip localizado). Entrada = pose+pelota (no vídeo crudo; vídeo crudo
queda como extensión futura). La heurística ADR-0012 se conserva como baseline
comparativo (heurística vs aprendido). Detalle de la arquitectura de BST leído:
TCN(pose)+TCN(pelota) → transformer temporal por stream → cross-attention (Q=pose
del golpeador, K,V=pelota) → transformer de interacción → cabeza; variantes de
1v1 (Clean Gate, Aim Player) NO aplican a pádel 2v2, nos quedamos con el núcleo.

## Entorno

- WSL2 + RTX 3090. Crashes esporádicos de WSL ("catastrophic failure"):
  mitigados subiendo el límite de RAM de WSL de 16 a 20 GB (.wslconfig) y
  con hábito de commits frecuentes + entrenamientos reanudables.
- Inferencia pose+tracking a 1920: ~35 fps (más rápido que tiempo real).
  Entrenamiento court detector (60 ep, 1280): ~16 min; a 1920: ~40 min.

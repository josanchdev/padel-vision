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

## Entorno

- WSL2 + RTX 3090. Crashes esporádicos de WSL ("catastrophic failure"):
  mitigados subiendo el límite de RAM de WSL de 16 a 20 GB (.wslconfig) y
  con hábito de commits frecuentes + entrenamientos reanudables.
- Inferencia pose+tracking a 1920: ~35 fps (más rápido que tiempo real).
  Entrenamiento court detector (60 ep, 1280): ~16 min; a 1920: ~40 min.

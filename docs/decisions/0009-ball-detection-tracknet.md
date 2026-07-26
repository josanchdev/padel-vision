# ADR-0009: Detección de pelota y botes (Nivel 3)

**Estado:** aceptada · 2026-07-26

## Contexto

El Nivel 3 añade la pelota: su posición por frame, su trayectoria en coordenadas
de pista (vía la homografía ya disponible) y la detección de botes. Sobre esto se
construyen las stats de RESULTADO ("J4 falló el 60% de reveses"), que necesitan
saber si la pelota entró, salió o fue a la red — algo que el esqueleto del Nivel 2
no puede decir.

La pelota es el objeto más difícil de detectar: mide ~8×8 px en 1080p (medido en
las anotaciones de PadelTracker100), va con motion blur, y se ocluye tras jugadores
o cristal. Un detector frame-a-frame estándar la pierde. El estado del arte en
deportes de raqueta (TrackNet y sucesores) mira **3 frames consecutivos** y predice
un **mapa de calor**, usando el movimiento entre frames como señal para separar la
pelota diminuta del fondo estático.

## Decisiones

- **Arquitectura (Decisión A):** comparar **TrackNetV2 (2020, baseline citable)**
  vs **TrackNetV3 (2023, moderna)**, ambos entrenados con nuestro dataset y el mismo
  protocolo de evaluación. V3 añade corrección de trayectoria e *inpainting* de
  oclusiones; la hipótesis es que gane precisamente en los frames `occluded`. La
  comparativa con métricas y plots propios es la contribución científica —
  evidencia reproducible, no un *trust-me* con números ajenos. Se entrena el
  baseline entero (barato: U-Net ligera) porque en la defensa "entrené ambos con mi
  protocolo" vale infinitamente más que "el paper dice que V3 es mejor".
- **Datos como kickstarter (Decisión B):** las ~37.900 cajas de pelota de
  PadelTracker100 entrenan el modelo (igual que el dataset de vehículos de noche en
  Ferrovial dio la v1). El modelo resultante es propio y en inferencia detecta la
  pelota **solo con sus pesos** — nunca consume las anotaciones del dataset. Respeta
  el principio rector (ADR-0005, [[feedback-no-dataset-dependency]]).
- **Conversión de formato (Decisión C):** las cajas COCO `[x,y,w,h]` no sirven
  directamente para TrackNet, que entrena con **heatmaps gaussianos** (una gaussiana
  2D, σ≈2-3 px, centrada en la pelota). Convertimos centro de caja → heatmap. Se
  descartó entrenar un YOLO de objeto diminuto con las cajas sin convertir: eso es
  justo el paradigma frame-a-frame que TrackNet supera. La conversión se documenta
  como "reto de datos resuelto" para la memoria.
- **Alcance (Decisión D):** pelota 2D en imagen + trayectoria en coords de pista +
  botes. **NO** se persigue reconstrucción 3D tipo Hawk-Eye: es inviable con vídeo
  monocular y sería una promesa deshonesta ante el tribunal.
- **Métricas y evidencia (Decisión E):** se introduce **MLflow** de verdad en el
  repo (estaba en el stack sin uso) para registrar métricas y curvas de ambos
  entrenamientos, MÁS export de **PNG/HTML autocontenidos** (curvas train/val,
  precisión de posición en px y en metros tras homografía) listos para las slides
  de la defensa.
- **Botes (Decisión F):** un bote es un cambio de dirección de la trayectoria de la
  pelota (en coords de pista) que NO coincide con un golpe de jugador. Las
  categorías de evento del propio dataset (`shot-event`, `serve-event`, ...) dan el
  frame exacto de cada golpe — se usan **offline** para validar el detector de botes,
  no en inferencia.

## Consecuencias

- (+) La pelota desbloquea las stats de resultado, el "producto" que la web exhibe.
- (+) Segunda comparativa baseline-vs-moderno propia (tras ST-GCN vs PoseConv3D del
  Nivel 2): refuerza el patrón metodológico del TFG.
- (+) MLflow entra en el repo con un caso de uso real y trazable.
- (−) La pelota está anotada en 35-82% de frames según partido; los tramos sin
  anotación no aportan supervisión (pero el modelo interpola en inferencia).
- (−) La detección de botes con vídeo monocular es aproximada (sin profundidad real);
  se valida contra los eventos de golpe del dataset y se reporta su fiabilidad con
  honestidad.

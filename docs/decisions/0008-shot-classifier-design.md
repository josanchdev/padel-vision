# ADR-0008: Diseño del clasificador de golpes (Nivel 2)

**Estado:** aceptada · 2026-07-26

## Contexto

El Nivel 2 sustituye el dummy de golpes (heurística de muñeca, recall 0,94 / precisión 0,28) por un clasificador serio basado en esqueleto. Hay que fijar de qué datos aprende, qué clases predice, cómo se atribuye cada golpe a un jugador y qué arquitecturas se comparan. PadelTracker100 aporta pose, pelota y etiquetas de golpe por frame, pero NO indica qué jugador golpea.

## Decisiones

- **Atribución de golpe a jugador (Decisión 0):** en el instante del golpe, el jugador cuya muñeca está más cerca de la pelota anotada es quien golpea. Reutiliza la pelota del dataset; más fiable que heurísticas de movimiento.
- **Fuente de esqueletos (Decisión 1):** entrenar con poses de *nuestro* YOLO26-pose (corrido sobre los vídeos WPT), no con el GT de ViTPose. Motivo: eliminar el domain gap — el clasificador aprende sobre el mismo tipo de pose (ruidosa) que verá en inferencia. El GT queda como experimento de referencia ("cuánto se pierde por usar poses automáticas").
- **Taxonomía (Decisión 3):** 5 clases — derecha, revés, remate, saque, otro. La dejada (solo 25 ejemplos en la final F) se absorbe en "otro"; es inentrenable como clase propia. Recuperable cuando haya datos propios etiquetados.
- **Desbalanceo:** class weights (o focal loss) en vez de descartar clases minoritarias.
- **Arquitecturas (Decisión 2):** ST-GCN (2018) como baseline citable vs PoseConv3D (heatmaps 3D + CNN 3D, más robusto al ruido de keypoints) como moderna. La comparación con métricas por clase y matrices de confusión es la contribución científica.

## Diseño de los clips (windowing, decidido al construir)

- **Ventana de 32 frames centrada en el impacto** (~1 s a 30 fps): más ancha
  que el golpe medio (17 frames) para que el modelo vea preparación y
  acompañamiento, no solo el impacto. Potencia de 2 por comodidad.
- **Seguimiento por track id**: la atribución da el track del que golpea; se
  extrae su esqueleto en toda la ventana siguiendo ese id. Huecos (oclusión)
  se rellenan con la última pose válida (hold-fill); si falta en más de media
  ventana, el clip se descarta.
- **Normalización invariante**: cada esqueleto se centra en el punto medio de
  caderas y se escala por la longitud del torso. Una derecha en fondo cercano
  y otra en lejano quedan idénticas para el modelo (invarianza a
  posición/tamaño/cámara).
- **Clips negativos "NoShot"**: ventanas de rally lejos de cualquier golpe
  (>45 frames), para que el clasificador rechace movimientos que no son golpe
  — ataca directamente los falsos positivos del dummy. 6 clases finales.

## Consecuencias

- (+) El clasificador entrena y evalúa sobre datos representativos del despliegue real.
- (+) La atribución por pelota conecta dos anotaciones del dataset y da clips por jugador.
- (+) Varios experimentos publicables: GT vs YOLO, ST-GCN vs PoseConv3D, efecto del desbalanceo.
- (−) La atribución depende de la pelota, que no está en todos los frames (35-82% según partido); golpes sin pelota visible se descartan del entrenamiento.
- (−) Empezar con 5 clases deja la dejada fuera hasta tener más datos.

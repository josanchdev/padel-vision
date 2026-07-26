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

## Consecuencias

- (+) El clasificador entrena y evalúa sobre datos representativos del despliegue real.
- (+) La atribución por pelota conecta dos anotaciones del dataset y da clips por jugador.
- (+) Varios experimentos publicables: GT vs YOLO, ST-GCN vs PoseConv3D, efecto del desbalanceo.
- (−) La atribución depende de la pelota, que no está en todos los frames (35-82% según partido); golpes sin pelota visible se descartan del entrenamiento.
- (−) Empezar con 5 clases deja la dejada fuera hasta tener más datos.

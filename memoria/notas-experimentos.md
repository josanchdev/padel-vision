# Notas para el capítulo de Experimentos

Cosas que han salido al escribir el capítulo 4 y que se decidió dejar para Experimentos.
No es texto de la memoria: es una lista de lo que hay medido, dónde está y con qué cautelas.
Se va ampliando según se escribe.

## 1. Tamaño de entrada del modelo de pose (§4.2.2)

- **Qué se dice en 4.2.2:** por defecto el modelo reduce el fotograma a 640 px; con ese valor los
  jugadores del fondo no se detectaban; se subió a 1920 px (ver Experimentos).
- **Dato:** `docs/metrics/pose_resolution.json`, script `scripts/evidence_pose_resolution.py`.
  11 puntos (el «01» de cada torneo), 1 de cada 4 fotogramas, umbral y máscara del sistema.

  | Tamaño | Cuatro jugadores | Pareja del fondo | ms/fotograma |
  |---|---|---|---|
  | 640 | 0 % | 0 % | 14,3 |
  | 960 | 6 % | 5 % | 14,9 |
  | 1280 | 40 % | 40 % | 16,5 |
  | 1920 | 94 % | 95 % | 16,9 |

- **Cautelas:** decirlo como «en las pruebas que hicimos», no como ley general. No citar «60 px» para
  los jugadores del fondo (en un fotograma de Vigo medían 99 y 135 px; el comentario de `pose.py` es el
  caso peor, no una medida). El 640 por defecto está comprobado en Ultralytics 8.4.92.
- **Pendiente:** si se quiere la tabla en la memoria, generarla desde el JSON como las demás
  (patrón de `scripts/build_dataset_table.py`).

## 2. YOLO26n frente a YOLO26m (§4.2.2)

- **Dato:** solo un comentario en `scripts/extract_rally_features.py` (líneas ~70-74): con el
  nano aparecen los cuatro jugadores en el 85 % de los fotogramas de un punto, con el mediano en el
  55 %, y el nano es unas 2 veces más rápido. Explicación escrita allí: el cuello de botella son los
  jugadores pequeños del fondo, no la capacidad del modelo.
- **Cautelas:** no hay JSON ni script, y es un solo punto. Las variantes s, l y x **no** están medidas;
  solo se puede afirmar la comparación con la mediana.
- **Pendiente:** si se menciona, repetirlo con un script versionado (como el del punto 1, cambiando el
  modelo) para tener cifra y respaldo.

## 3. Umbral de confianza de la pose: 0,25 y no 0,4 (§4.2.2)

- **Dato:** comentario en `scripts/extract_rally_features.py`: con 0,25 aparecen los cuatro jugadores
  en el 85 % de los fotogramas, con 0,4 en el 80 %. `analyze_rally` usa 0,25 (`POSE_CONFIDENCE`); el
  valor por defecto de la clase `PoseDetector` (0,4) no es el del sistema.
- **Razón añadida:** el clasificador de golpes se entrenó con poses extraídas con 0,25, así que en
  inferencia hay que producirlas igual.
- **Cautelas:** igual que el punto 2: un punto y sin script.

## 4. ByteTrack (§4.2.2)

- **Configuración:** `packages/cv/src/padel_cv/trackers/padel_bytetrack.yaml` (`match_thresh` 0,95,
  `track_buffer` 120, `track_high_thresh` 0,25, `track_low_thresh` 0,1, `new_track_thresh` 0,4).
- **Dato histórico:** `docs/experiments.md` («Tracking e identidad»): con la configuración propia,
  4 de 4 jugadores estables en un punto completo (407 fotogramas); con la de serie, saltos de
  identidad en los jugadores del fondo. Es una prueba temprana, anterior a la identidad J1-J4 actual.
- **Pendiente de comprobar:** si la segunda asociación de ByteTrack (la de `track_low_thresh` 0,1,
  que aprovecha detecciones de baja confianza) llega a actuar, porque el detector ya descarta lo
  que está por debajo de 0,25. Hasta saberlo, no escribir que se aprovechan esas detecciones.
- **Si se quiere respaldo:** medir cambios de identidad con la configuración de serie y con la propia
  en varios puntos (hoy no hay script).

## 5. Homografía y pista marcada a mano (§4.2.2, ADR-0018)

- La cámara se mueve menos de 3 px en 10 de los 11 torneos; Valladolid sube 9,9 px desde el punto 04
  (`court_registry.py`).
- Detector propio de keypoints de pista: error de reproyección de ~0,11-0,15 m en 6 torneos, deriva de
  1-5 m en 3 y fallo en 2 (ADR-0018). El residuo de 0,113 m de la homografía marcada a mano mide
  consistencia, no exactitud.
- **Pendiente:** comprobar estas cifras contra el ADR-0018 antes de citarlas.

## 6. Coste de procesado (más probable en §4.4 que en Experimentos)

- `docs/metrics/processing_time.json`: 67,5 s para 64,3 s de vídeo (1,05 veces el tiempo real) en una
  RTX 3090; la pose son 27,1 s. **No** escribir «más rápido que tiempo real»: eso es del detector solo.

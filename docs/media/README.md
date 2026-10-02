# Capturas e imágenes

Material gráfico para la memoria. Las figuras de resultados están aparte, en
`docs/metrics/figures/`, y las regenera su script; estas son capturas.

## El sistema final

| Imagen | Qué muestra |
|---|---|
| `web_portada.png` | Portada del visor web: los 12 puntos exportados (1920×1080, tema claro) |
| `web_punto_en_vivo.png` | Vista de un punto (VIGO_11, segundo 12,4, golpe de J1: derecha, 94 %): vídeo anotado, pista en vivo, línea de tiempo por jugador y tarjetas |
| `web_punto_mapa_de_calor.png` | El mismo instante con el panel de pista en mapa de calor |
| `video_anotado.jpg` | Frame del vídeo anotado del mismo punto, a resolución completa: esqueletos, identidad J1-J4, pelota y etiqueta del golpe |
| `Screenshot-interfaz-tag-golpes.png` | Herramienta propia de etiquetado del tipo de golpe (`padel-cv annotate-types`) |
| `tool_puntos_campo.png` | Herramienta propia de marcado de la pista, 6 clics (`padel-cv annotate-court`) |

Las capturas de la web se hicieron con Chromium sin pantalla sobre los puntos
exportados con los modelos finales (`scripts/export_points.py`); el frame, con
OpenCV sobre el vídeo exportado (frame 310).

## Exploración (histórico)

| Imagen | Qué muestra |
|---|---|
| `audio_energia_global.png`, `audio_highpass_3khz.png` | Primeras pruebas de detectar el golpe por la energía del audio (julio–agosto) |
| `senal_por_frame_picos.png`, `senal_por_ventana_meseta.png` | Picos frente a ventanas en la salida del detector de audio (ADR-0015 D3) |
| `generalization_santiago_1080p_a.jpg`, `generalization_santiago_1080p_b.jpg` | Prueba de generalización del pipeline antiguo en un vídeo de otra competición (Santiago) |

# ADR-0017: La web es un visor estático, sin backend

**Estado:** aceptada · 2026-09-28 · **supera ADR-0007** (cola Arq) y la parte de
ADR-0010/0011 en que la API servía los datos y el frontend

## Contexto

La web subía vídeos y los procesaba con FastAPI + cola Arq + Redis + worker en
Docker, y ejecutaba el pipeline antiguo. Con el sistema cerrado, la web solo
tiene que **mostrar** puntos ya analizados, y todo ese backend existía para
procesar, no para mostrar.

## Decisión

- `scripts/export_points.py` procesa cada vídeo con `analyze_rally` (el mismo
  código que las evaluaciones) y escribe una carpeta por punto en
  `packages/web/public/points/<id>/`: `point.json`, vídeo anotado 1080p con
  sonido, miniatura y vista previa; más un `index.json` para la portada. El
  contrato de datos está en `padel_ml.web_export`.
- La web (React + Vite + Motion) lee esos ficheros. Dos pantallas por hash
  (`#/` portada, `#/p/<id>` punto), sin servidor propio.
- Diseño decidido con maquetas y datos reales: portada en cuadrícula de 4, vista
  de punto en «cabina» (vídeo + pista en vivo/mapa de calor + línea de tiempo por
  jugador + tarjetas), color = jugador y forma = tipo de golpe, y la cabina
  ajustada al alto de pantalla.

## Consecuencias

- Añadir un punto es un comando de terminal, no un formulario.
- Los puntos exportados no van a git (vídeos pesados, regenerables).
- FastAPI, Arq, Redis, el worker y casi todo Docker quedan sin uso: se eliminan
  en la limpieza global.

# CVAT — herramienta de anotación

CVAT corre entero en Docker: no instala nada en el proyecto ni en el sistema. Arrancar, etiquetar, parar. Las anotaciones viven en volúmenes de Docker y sobreviven a `down` y a reinicios.

## Requisito

Docker Desktop para Windows con backend WSL2 (Settings → Resources → WSL integration → activar para esta distro). Una vez instalado, `docker` funciona desde esta terminal WSL.

## Uso

```bash
./cvat.sh up      # arranca → http://localhost:8080
./cvat.sh user    # crear usuario admin (solo la primera vez)
./cvat.sh down    # parar y liberar recursos (las anotaciones SE CONSERVAN)
./cvat.sh status  # ver contenedores
```

## Esquema de anotación de pista (13 keypoints, ver ADR-0001)

Crear en CVAT una label `court` de tipo *skeleton* con estos 13 puntos **en este orden exacto** — el orden define los índices que espera el modelo (`padel_cv/court.py::COURT_KEYPOINT_NAMES`). Todos sobre el SUELO — nunca cristal ni postes:

| # | Nombre | Dónde está |
|---|--------|------------|
| 0 | corner_near_left | Esquina suelo del fondo cercano, izquierda |
| 1 | corner_near_right | Esquina suelo del fondo cercano, derecha |
| 2 | service_near_left | Línea de servicio cercana × pared izquierda |
| 3 | service_near_center | T central: línea central × servicio cercana |
| 4 | service_near_right | Línea de servicio cercana × pared derecha |
| 5 | net_left | Línea de red en el suelo × pared izquierda |
| 6 | net_center | Centro de pista bajo la red |
| 7 | net_right | Línea de red en el suelo × pared derecha |
| 8 | service_far_left | Línea de servicio lejana × pared izquierda |
| 9 | service_far_center | T central: línea central × servicio lejana |
| 10 | service_far_right | Línea de servicio lejana × pared derecha |
| 11 | corner_far_left | Esquina suelo del fondo lejano, izquierda |
| 12 | corner_far_right | Esquina suelo del fondo lejano, derecha |

"near/far" e izquierda/derecha son relativos a la cámara de la imagen que anotas.

Punto no visible en la imagen → marcarlo `outside`; visible pero tapado (jugador delante) → `occluded` colocándolo donde estaría. Precisión ante todo: usa zoom — 2-3 px de error en la mitad lejana son ~10 cm reales.

## Flujo completo

1. `uv run padel-cv sample-frames <dir_videos> -o data/annotation/<batch> --per-video N`
2. Crear task en CVAT subiendo esa carpeta de imágenes.
3. Anotar (~30-60 s por imagen).
4. Exportar como **COCO Keypoints 1.0** a `data/annotation/<batch>_export/`.

La carpeta `cvat-src/` (clon del código oficial) está en `.gitignore`; se descarga sola en el primer `./cvat.sh up`.

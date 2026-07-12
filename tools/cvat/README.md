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

Crear en CVAT una label `court` de tipo *skeleton* con estos 13 puntos (todos sobre el SUELO — nunca cristal ni postes):

| Punto | Descripción |
|-------|-------------|
| A1, A2 | Esquinas suelo del fondo cercano (izq, dcha) |
| S1, S2 | Línea de servicio cercana × paredes (izq, dcha) |
| T1 | T central: línea central × línea de servicio cercana |
| N1, N2 | Línea de red en el suelo × paredes (izq, dcha) |
| C | Centro de pista bajo la red |
| T2 | T central: línea central × línea de servicio lejana |
| S3, S4 | Línea de servicio lejana × paredes (izq, dcha) |
| B1, B2 | Esquinas suelo del fondo lejano (izq, dcha) |

Punto no visible en la imagen → marcarlo `outside`; visible pero tapado (jugador delante) → `occluded` colocándolo donde estaría.

La carpeta `cvat-src/` (clon del código oficial) está en `.gitignore`; se descarga sola en el primer `./cvat.sh up`.

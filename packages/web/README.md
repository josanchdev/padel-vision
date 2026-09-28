# padel-web

Visor de Padel Vision (ADR-0017). React + Vite + TypeScript + Motion.

Es una web **estática**: no procesa vídeos ni habla con ninguna API. Muestra los
puntos que el sistema ya ha analizado, que son ficheros en `public/points/`.

## Añadir puntos

Se procesan desde la raíz del repo; cada vídeo tarda algo más de un minuto:

```bash
uv run python scripts/export_points.py data/raw/.../20230528_VIGO_11.mp4 [más vídeos]
```

Deja en `public/points/<id>/` el `point.json` (golpes y posiciones), el vídeo
anotado, la miniatura y la vista previa, y actualiza `public/points/index.json`.
Esa carpeta no va a git.

## Desarrollo

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # a dist/, con los puntos incluidos
```

## Estructura

- `pages/Home` — la cuadrícula de puntos, con vista previa al pasar el ratón.
- `pages/PointView` — la cabina de un punto: todo sigue el reloj del vídeo
  (`useVideoTime`).
- `components/` — `CourtPanel` (pista en vivo y mapa de calor), `ShotTimeline`
  (un carril por jugador), `PlayerCard`, `ShotShape` (la forma de cada tipo).
- `data.ts` — el contrato de ficheros; `court.ts` — la geometría y la
  orientación de la pista; `heatmap.ts`; `play.ts` — colores, tipos y formatos.

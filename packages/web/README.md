# padel-web

Dashboard de Padel Vision. React + Vite + TypeScript + Motion (ADR-0011).

El producto no es el vídeo procesado (eso es verificación visual): son los datos
estructurados para tomar decisiones. Esta web los explora — tabla de golpes
filtrable, clip de cada golpe, KPIs — consumiendo la API (`/matches`, `/data`,
`/clip`).

## Desarrollo

```bash
npm install
npm run dev      # dev server con proxy a la API en :8000
```

Necesita la API corriendo (`tools/dev.sh` o `docker compose up`) para que el
proxy sirva `/matches`, `/data`, `/clip`.

## Build

```bash
npm run build    # tsc + vite; salida a ../api/src/padel_api/static (la sirve FastAPI)
```

El bundle compilado es artefacto generado (gitignoreado); en Docker lo produce
un stage de Node y se copia al `static/` de la API.

## Estructura

- `src/api.ts` — cliente tipado de la API (la capa que sobrevive rediseños).
- `src/theme.css` — tokens de diseño (identidad como tokens: el rojo URJC / logo
  se cambian aquí sin tocar componentes).
- `src/App.tsx` — layout, subida, lista de partidos, vista de detalle.
- `src/ShotTable.tsx` — tabla de golpes con filtros.
- `src/ClipModal.tsx` — modal de clip con marcador del golpe exacto.

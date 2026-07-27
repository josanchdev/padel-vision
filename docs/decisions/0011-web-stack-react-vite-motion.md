# ADR-0011: Stack web — React + Vite + Motion, build estático servido por FastAPI

**Estado:** aceptada · 2026-07-27

## Contexto

La web es el escaparate del producto y Jorge la considera importantísima: debe
sorprender (estándares 2026), no ser un "generic AI website". El mockup actual
(`static/index.html` a mano) validó el bucle diseño→backend→frontend, pero su
piel es deliberadamente desechable. El factor "wow" real (animaciones fluidas a
60fps, transiciones, micro-interacciones con propósito) no sale de CSS/JS a mano:
sale de frameworks modernos sobre React. Además conviene montar la base bien para
expandir sin rehacer cada sección. Investigación de referencias y stack en
`docs/web-design-dossier.md`.

## Decisiones

- **Framework (Decisión A):** **React + Vite**. Componentes reutilizables (una
  sección se diseña una vez y se reusa en KPIs, heatmap, timeline, ...), build
  rápido, base expandible. Vite por velocidad de dev y simplicidad frente a
  alternativas.
- **Animación (Decisión B):** **Motion (ex-Framer Motion, `motion/react`)** como
  librería principal. API declarativa que entiende el ciclo de vida de React
  (enter/exit, layout, gestos) — lo recomendado para UI de dashboard en 2026,
  ~30 KB. **GSAP** se añade PUNTUALMENTE solo si una sección pide coreografía
  compleja (scroll-scenes, secuencias al milisegundo); no se adopta de base.
- **Despliegue (Decisión C):** **build estático servido por FastAPI**. Vite
  compila a HTML/JS/CSS estáticos que FastAPI sirve (como hoy sirve
  `index.html`). Un solo servicio, un solo puerto, encaja con el Docker actual
  sin añadir contenedores. En desarrollo se usa el dev server de Vite con proxy a
  la API. Se descarta un servicio web separado por sobredimensionado para un TFG.
- **Reutilización:** la capa `api{}` del mockup (fetch a `/data`, `/clip`,
  `/matches`) se porta tal cual; la lógica de conexión no se tira, solo cambia la
  piel — como se diseñó a propósito.
- **Identidad:** rojo URJC como acento, implementado como **tokens** (variables
  de tema) para poder cambiar logo/paleta tras la reunión con el tutor sin
  rehacer componentes.

## Consecuencias

- (+) Base moderna y expandible: nuevas secciones nacen con el lenguaje visual ya
  definido, sin rehacer.
- (+) Animación de calidad (60fps) con propósito, no decorativa.
- (+) Un solo contenedor/puerto: el `docker-compose` actual apenas cambia (sirve
  el bundle en vez del html suelto).
- (−) Mete build (Vite) y stack JS en `packages/web` (hoy vacío/andamio); hay que
  configurar el pipeline de build y su copia a la imagen Docker.
- (−) Más superficie que mantener; justificable porque el producto ES la interfaz
  de decisión y merece un stack serio (defendible ante el tribunal).
- (−) El primer montaje es andamiaje (config Vite, estructura de componentes)
  antes de ver resultado visual; se hace una vez.

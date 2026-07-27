# Dossier: qué hace memorable una web en 2026 + stack

Material de referencia para el rediseño "wow" de la plataforma (opción C).
Objetivo: NO caer en el "generic AI website" (correcto pero olvidable) y montar
la base técnica bien para expandir sin rehacer. Fuentes al final.

## Qué hace memorable una web en 2026 (los principios)

Destilado de sitios premiados (Awwwards, FWA, CSS Design Awards) y análisis de
tendencias 2026:

1. **La tipografía ES el diseño, no va encima.** Titulares enormes, fuentes
   variables, letras cinéticas que reaccionan al scroll. El texto grande es el
   protagonista visual, no un adorno. (Encaja con lo que Jorge intuía:
   "textos y fuentes grandes".)
2. **60 fps o no cuenta.** "Beauty at 60fps is the whole discipline." Una
   animación que va a tirones resta en vez de sumar. Gran diseño 2026 = gran
   ingeniería. Esto condiciona el stack (ver abajo).
3. **Se acabó la animación por la animación.** Lo premiado ya no es "flashy":
   es interacción que se siente BIEN, que responde al usuario. Movimiento con
   propósito (feedback, continuidad, jerarquía), no decorativo.
4. **Profundidad e inmersión.** Scroll-triggered, 3D/WebGL donde aporta,
   modelos interactivos. Para NOSOTROS esto se traduce en: los datos como
   espectáculo (un heatmap de pista a pantalla completa, una trayectoria
   animada) más que tablas planas.
5. **Espacio radical + jerarquía brutal.** Una cosa domina cada pantalla, el
   resto se subordina. El aire generoso se lee como premium y confianza.

**Traducción a Padel Vision:** somos una herramienta de decisión densa en datos,
NO una landing de marketing. Tomamos de lo anterior: tipografía fuerte, datos
como protagonistas (números grandes, heatmaps a lo grande), micro-interacciones
con propósito (hover en una fila resalta al jugador, transición al abrir el
clip), 60fps. Evitamos: 3D gratuito, "dopamine design" saturado, animación
decorativa que distrae del análisis. Identidad = rojo URJC como acento (pendiente
del tutor, como tokens intercambiables).

## Stack técnico (decisión de arquitectura → ADR)

Jorge decidió (con criterio): el "wow" real no sale de CSS a mano, sale de
frameworks modernos sobre React, y conviene montar la base bien para expandir.

**Recomendación de stack (a confirmar en ADR):**

- **React + Vite** — componentes reutilizables (una sección se diseña una vez y
  se reusa), build rápido, base sólida para crecer. Vite sobre CRA/otros por
  velocidad de dev y simplicidad.
- **Motion (ex-Framer Motion)** como librería de animación PRINCIPAL. Motivo:
  API declarativa que entiende el ciclo de vida de React (enter/exit, layout,
  gestos hover/drag/tap) — ideal para UI de dashboard. ~30 KB. Es lo recomendado
  para "application UI" en 2026. (Rebautizada de Framer Motion a Motion en 2025;
  import `motion/react`.)
- **GSAP** como opción SECUNDARIA, solo si una sección concreta necesita
  coreografía compleja (scroll-scenes, secuencias de decenas de elementos al
  milisegundo). Muchos sitios pro combinan: GSAP para secciones "marketing",
  Motion para la UI de la app. Para nosotros, que somos app, Motion cubre la
  mayoría; GSAP se añade puntualmente si hace falta un momento "hero".
- **Gráficos de datos:** a decidir en su momento (heatmap de pista, timeline).
  Candidatos: Recharts (simple, declarativo) o D3 (potente, control total) o
  canvas/WebGL para el heatmap si se quiere impacto. No urge ahora.

**Consecuencias a sopesar en el ADR:**
- (+) Base moderna, componentes reutilizables, animación de calidad, expandible.
- (+) La capa `api{}` del mockup actual (fetch a /data, /clip) se porta tal cual.
- (−) Mete build (Vite) y stack JS en `packages/web` (hoy es un index.html).
- (−) Integración con el Docker/API actual: hay que servir el bundle (build
  estático servido por FastAPI, o dev server separado). Decisión de despliegue.
- (−) Más superficie que mantener; para un TFG defendible hay que justificar por
  qué (respuesta: el producto ES la interfaz de decisión; merece stack serio).

## Referencias para inspirarse (llevar a la sesión de diseño)

- Awwwards / FWA / CSS Design Awards — "Site of the Day" (el estándar).
- By-Kin (estudio UK, 4 premios 2026) como ejemplo de ejecución.
- Para el TONO correcto (herramienta seria, no marketing): dashboards tipo
  Linear/Vercel por pulcritud; analítica deportiva (Hudl/Wyscout) por el
  lenguaje de componentes, PERO simplificado (no copiar su densidad comercial,
  ver [[web-sections-and-data-contract]]).

## Fuentes

- Web design 2026 trends: thesource.com, designrush.com, figma.com resource
  library, hontran.dev (WebGL/Awwwards).
- Stack animación: hontran.dev (GSAP vs Framer Motion), good-fella lab, satish
  kumar guide, pkgpulse (bundle sizes). Consenso: Motion para UI de app, GSAP
  para coreografía compleja.

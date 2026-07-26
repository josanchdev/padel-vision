# Secciones de la plataforma y contrato de datos

Documento puente entre la **capa de datos** (ADR-0010, lo que el pipeline
genera) y el **diseño de la web** (que se hará en Claude Design). Sirve para dos
cosas: (1) decidir qué secciones tiene la plataforma más allá del reproductor de
vídeo, y (2) para cada sección, fijar qué datos consume del esquema — lo que
revela qué falta añadir. Diseñar las secciones con datos concretos es lo que
evita el "generic AI website": se lleva a Claude Design contenido real, no un
vacío que la IA rellena con gradientes.

**Principio (insight de Jorge):** el vídeo procesado es *show*; el producto es la
capa de datos para tomar decisiones. Cada sección justifica su existencia por la
DECISIÓN que habilita, no por lo vistosa que queda.

## Qué datos existen hoy (`schema_version: 1`)

Fuente: salida real de `padel-cv process --data-out` (ADR-0010).

| Colección | Campos | Una fila es… |
|-----------|--------|--------------|
| `players` | frame_index, timestamp_s, player_id, track_id, court_x_m, court_y_m, on_court | posición de un jugador en un frame |
| `shots` | frame_index, timestamp_s, player_id, label, confidence | un golpe (tipo + quién + cuándo) |
| `ball` | frame_index, timestamp_s, image_x, image_y, court_x_m, court_y_m, confidence | la pelota en un frame |
| `bounces` | frame_index, image_x, image_y | un bote |

Metadatos: `source_video`, `fps`, `schema_version`.

## Secciones propuestas

Ordenadas por lo que ya es fiable HOY (arriba) hacia lo que depende de trabajo
futuro (abajo). Cada una: qué muestra, qué decisión habilita, qué datos consume,
y qué falta.

### 1. Resumen del partido (landing del análisis)

- **Muestra:** conteos de cabecera (nº de golpes por tipo, por jugador; nº de
  botes; duración). Tarjetas KPI.
- **Decisión:** "de un vistazo, ¿qué pasó en este partido?"
- **Datos:** agregación de `shots` (group by label, por player_id) y `bounces`.
- **Falta:** nada — se calcula de lo que hay. Es la sección más barata y de las
  más útiles.

### 2. Tabla de golpes (filtrable / exportable)

- **Muestra:** una fila por golpe: minuto, jugador, tipo, confianza. Filtros por
  jugador y tipo; orden; botón "exportar CSV".
- **Decisión:** "enséñame todos los reveses de J4" — el equivalente al CSV de
  Ferrovial pero interactivo. La sección más cercana al núcleo del producto.
- **Datos:** `shots` directo. El CSV ya se genera (ADR-0010).
- **Falta:** nada del dato; es UI sobre `shots`.

### 3. Mapa de calor de posiciones

- **Muestra:** silueta de la pista (top-down, ya tenemos el minimapa) con un
  heatmap de dónde estuvo cada jugador. Selector de jugador.
- **Decisión:** "¿J3 cubre bien su zona o se queda pegado a la pared?"
- **Datos:** `players` (court_x_m, court_y_m) filtrado por player_id. La
  geometría de pista ya está en `padel_cv.court`.
- **Falta:** nada del dato; es densidad 2D sobre posiciones existentes.

### 4. Trayectoria de la pelota

- **Muestra:** la pista con la traza de la pelota (court_x_m, court_y_m) y los
  botes marcados. Opcional: filtrar por tramo de tiempo.
- **Decisión:** "¿por dónde pasa la pelota?, ¿dónde botan?"
- **Datos:** `ball` + `bounces`.
- **Falta:** los botes hoy solo llevan posición en imagen (image_x/y), no en
  metros — habría que proyectarlos a court_x_m/y_m (mejora pequeña del schema).

### 5. Timeline de eventos

- **Muestra:** línea temporal del partido con marcas de golpes y botes; clic en
  una marca salta al vídeo en ese frame.
- **Decisión:** navegar el partido por eventos en vez de por tiempo bruto.
- **Datos:** `shots` + `bounces` con timestamp_s/frame_index.
- **Falta:** unir vídeo + timeline (la web necesita servir el vídeo y saltar a
  un frame). Depende de la sección de vídeo.

### 6. Reproductor de vídeo anotado (NO es la estrella)

- **Muestra:** el vídeo procesado. Verificación visual, no el producto.
- **Decisión:** "¿me fío de lo que detectó?" — confianza en los datos.
- **Datos:** el mp4 de `padel-cv process`.
- **Falta:** nada; ya existe. Deliberadamente NO es la sección principal.

### 7. [Futuro] Stats por jugador / resultado de puntos

- **Muestra:** "J4 falló el 60% de reveses", puntos ganados/perdidos, winners.
- **Decisión:** la explotación de alto nivel.
- **Datos:** requiere la pila de robustez (cámara-en-pista → segmentación de
  puntos → winner/error), ver `docs/backlog.md`. NO comprometido.
- **Falta:** todo el nivel 5-6 de la pila. Sección "en construcción" hasta
  entonces.

## Qué añadir al schema (descubierto al diseñar)

- **Botes en metros:** proyectar `bounces` a court_x_m/y_m (sección 4).
- **Bloque de resumen precomputado (opcional):** para no recalcular agregados en
  el cliente, el JSON podría traer un `summary` con conteos. Decisión abierta:
  ¿lo precomputa el pipeline o lo agrega la web? (recomendación: la web agrega;
  mantiene el schema como datos crudos y el resumen como vista).
- **Flags de robustez (futuro):** cuando exista, `players`/frames llevarán
  `camera_on_court` y `point_id`, como columnas más — encaja sin rediseño.

## Referencia visual: cómo decidirla (para Claude Design)

Jorge aún no ha fijado la referencia. Marco para decidir sin caer en genérico:

**El producto es una herramienta de decisión densa en datos, para un equipo
técnico (entrenadores URJC), no una app de consumo.** Eso descarta el look
"landing SaaS alegre" y apunta a **dashboards de analítica profesional**. Dos
familias:

- **Analítica deportiva pro** (Hudl, Wyscout, Second Spectrum): timelines,
  heatmaps, tablas densas, tono serio. Máxima afinidad temática — es
  literalmente el mismo dominio. Riesgo: pueden verse "aburridos"/corporativos.
- **Dashboards SaaS modernos** (Linear, Vercel): limpios, tipografía fuerte,
  espacio, componentes nítidos. Se sienten "2026" y premium. Riesgo: menos
  específicos de deporte, hay que aportar el carácter con datos y marca.

**Recomendación:** una síntesis — el **rigor y los componentes de la analítica
deportiva** (heatmaps, timelines, tablas densas de verdad) con la **pulcritud
tipográfica y el espaciado de un SaaS moderno**. Marca URJC como acento (paleta
por decidir: rojo institucional URJC vs verde pádel — abierto). Lo que hace que
NO sea genérico no es el estilo elegido, sino que las pantallas están llenas de
**datos reales de un partido concreto** con densidad de herramienta profesional.

Material a llevar a Claude Design: este inventario + un export JSON real + la
decisión de paleta.

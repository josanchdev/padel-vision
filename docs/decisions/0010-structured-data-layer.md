# ADR-0010: Capa de datos estructurados (el producto real)

**Estado:** aceptada · 2026-07-27

## Contexto

El pipeline genera mucha información por frame (poses, identidades, posiciones en
metros, golpes, pelota, botes) pero hoy la **tira**: solo la pinta sobre el vídeo
y la pierde. Un vídeo anotado es *verificación visual*, no un producto — un
entrenador no puede filtrar, comparar ni agregar sobre píxeles. El valor está en
los **datos estructurados** que permiten tomar decisiones (como el CSV de
detecciones era el entregable real en la experiencia previa de Jorge en
Ferrovial, no el vídeo con cajas).

Este ADR reorienta el producto: la salida canónica del sistema es una **capa de
datos consultable**; el vídeo pasa a ser una vista más, no la salida. Es la base
sobre la que se construye cualquier sección de la web que no sea el reproductor,
y el contrato que la API expone.

## Decisiones

- **Registro atómico (Decisión A):** el pipeline emite, por partido, registros
  estructurados: detecciones por frame (jugador, track_id, posición en pista m,
  on_court), golpes (frame, jugador, tipo, confianza), botes (frame, posición en
  pista m) y pelota (frame, posición imagen, posición pista m, confianza).
- **Formato (Decisión B):** **JSON canónico** (rico, anidado, lo que la API/web
  consumen nativo) + **export CSV** derivado (una fila por evento, lo que un
  entrenador abre en Excel). Un mismo origen, dos vistas.
- **Separación datos/vídeo (Decisión C):** la extracción de datos se separa del
  renderizado de vídeo. Habrá un modo "solo datos" (sin escribir vídeo, más
  rápido) para cuando no se necesite la verificación visual. Hoy ambos están
  pegados en el CLI; se desacoplan.
- **Esquema versionado (Decisión D):** la salida lleva un `schema_version`. El
  formato es un **contrato**: versionarlo deja que la web sepa qué esperar y que
  el esquema evolucione sin romper clientes. Es lo que lo hace producto y no un
  volcado.

## Consecuencias

- (+) La web puede tener secciones reales (tabla de golpes filtrable, mapa de
  calor, timeline) porque consume datos, no un vídeo.
- (+) Modo "solo datos" acelera el procesamiento cuando no hace falta vídeo.
- (+) El contrato versionado permite prototipar la UI en paralelo (descubrir qué
  datos necesita cada sección retroalimenta este esquema).
- (+) Encaja la futura pila de robustez (cámara-en-pista, punto en juego) como
  columnas más del registro, sin rediseñar nada.
- (−) Hay que refactorizar el CLI para separar extracción de renderizado.
- (−) El esquema versionado añade disciplina (migrar al cambiarlo), coste que se
  paga por tratar la salida como contrato.

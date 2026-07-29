# ADR-0012: Detección de golpes por la pelota (no por pico de muñeca)

**Estado:** aceptada · 2026-07-29

## Contexto

La detección de golpes en inferencia (ADR-0008) usa dos etapas: (1) un pico de
velocidad de muñeca por jugador propone candidatos, (2) el clasificador
PoseConv3D dice el tipo. Validando sobre un partido real se vio que la etapa 1 es
débil: dispara con cualquier gesto brusco del brazo aunque no se golpee la pelota
(falsos positivos) y marca el golpe antes del impacto real (timing malo).

Ahora que el detector de pelota propio funciona bien (~96% de frames, conf ~0,97
en pista azul), se puede detectar el golpe por su causa física real: **la pelota
cambia bruscamente de dirección Y hay una muñeca de jugador cerca en ese
instante**. Esto reemplaza la heurística de muñeca por una señal fundamentada.

Nota de independencia (ADR-0005): la pelota se detecta con NUESTRO modelo
TrackNet en inferencia, no con anotaciones externas. El sistema sigue autónomo.

## Decisiones

- **Señal de impacto (Decisión A):** un golpe = cambio de dirección de la
  trayectoria de la pelota + una muñeca de jugador cerca de la pelota en ese
  frame. Combina las dos señales: descarta cambios sin jugador (botes) y gestos
  sin pelota (ruido). El jugador que golpea es el de la muñeca más cercana.
- **Sustitución (Decisión B):** esta detección reemplaza el pico de muñeca como
  etapa 1. La etapa 2 (PoseConv3D clasifica el tipo) se mantiene igual — recibe
  golpes bien detectados en vez de picos ruidosos.
- **Unificación bote/golpe (Decisión C):** bote y golpe comparten la señal base
  (cambio de dirección de la pelota). Un solo detector encuentra todos los
  cambios de dirección; cada uno se clasifica como **golpe** (muñeca cerca) o
  **bote** (sin jugador cerca). Elimina la duplicación con `bounces.py`; la
  lógica queda coherente.
- **Validación (Decisión D):** doble. **Métrica**: comparar detección por muñeca
  vs por pelota contra los frames de golpe anotados en PadelTracker100
  (`shot-event`), con precisión/recall — evidencia objetiva. **Visual**: procesar
  el partido real y ver que desaparecen los falsos positivos observados.

## Consecuencias

- (+) Ataca de raíz los falsos positivos y el mal timing (la queja principal).
- (+) Unifica bote y golpe: una sola lógica de "cambio de dirección", más limpia.
- (+) Físicamente fundamentado y defendible; comparativa muñeca-vs-pelota como
  material de memoria.
- (+) El clasificador de tipo (PoseConv3D) se reutiliza sin cambios.
- (−) Depende de que la pelota se detecte en el frame del impacto; si la pelota
  falla (pista nueva, oclusión), se pierde ese golpe. Mitigación futura: fallback
  a muñeca donde no haya pelota (no ahora, se evaluará).
- (−) El cambio de dirección en píxeles es ruidoso; hay que filtrar (prominencia)
  como en los botes.

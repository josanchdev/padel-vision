# Investigación: estado del arte en detección de golpes (feb 2026)

Dossier tras dar un paso atrás: el localizer casero (por-frame sobre pose+pelota)
da ~61 % recall / 70 % precisión (solo rallies) — insuficiente. Antes de seguir,
revisión del estado del arte para decidir el rumbo con evidencia.

## Hallazgo 1 — Nuestro problema tiene nombre: **Precise Event Spotting (PES)**

La literatura distingue tres tareas (survey arXiv 2505.03991):
- **Action Spotting (AS):** un keyframe por evento, tolerancia ancha (±50 frames).
- **Precise Event Spotting (PES):** precisión de frame (±0-2 frames). **ES LO
  NUESTRO** — un golpe es un evento instantáneo que hay que clavar al frame.
- **Temporal Action Localization (TAL):** intervalos con inicio/fin, para acciones
  largas. NO es lo nuestro.

Estábamos reinventando PES sin conocer sus métodos. Esto ya explica parte del fracaso.

## Hallazgo 2 — El SotA de PES NO es lo que montamos

Arquitectura ganadora (E2E-Spot, Hong 2022, repo `jhong93/spot`; T-DEED 2024;
mejoras 2025 en arXiv 2503.00147):

```
frames RGB → backbone compacto (RegNet-Y) + módulo de shift temporal (GSM/GSF)
           → Bi-GRU (contexto temporal largo) → clasificador por-frame → NMS
```

**Diferencias clave con nuestro enfoque:**
1. **Trabajan sobre RGB (píxeles), no sobre pose+pelota extraídas.** El backbone ve
   la imagen; capta el contacto pala-pelota directamente. Nosotros tiramos esa
   señal al reducir a keypoints.
2. **Bi-GRU sobre secuencias largas (128 frames)** para el contexto — "si hubo golpe
   solo se sabe mirando frames futuros". Nuestro TCN con ventana de 32 es corto.
3. **Módulos de shift temporal (GSM)** diseñados para captar el cambio frame-a-frame
   del instante del impacto.

**Nuestro BCE plano por-frame es exactamente lo que el survey dice que falla**: "la
cross-entropy estándar descuida los frames alrededor del evento". El desbalance
(evento raro entre muchos frames de fondo) es EL problema central de PES, y se
ataca con losses específicas, no con pos_weight.

## Hallazgo 3 — El AUDIO es la señal que nos falta (paper de PÁDEL, CVPR 2024)

"Multi-Modal Hit Detection and Positional Analysis in Padel Competitions"
(Decorte et al., CVPRW 2024, Ghent Univ.): **F1 92 % detectando golpes de pádel**
fusionando **AUDIO + pose**. La clave: **el "pop" de la pala al golpear es una señal
acústica limpísima y fácil de detectar** — mucho más que la geometría de pose/pelota
en la que nos hemos empeñado. Es exactamente nuestro caso (pádel, no bádminton) y
casi dobla nuestro F1. NO estamos usando el audio en absoluto.

## Hallazgo 4 — Losses/técnicas concretas que el SotA usa y nosotros no

- **Etiqueta temporal suave** (no un 1 duro en el impacto): ventana gaussiana
  alrededor del frame → el modelo aprende "cerca del golpe" gradualmente.
- **Context-aware loss** (Cioppa 2020): pondera frames por proximidad al evento
  (+12,8 % en SoccerNet).
- **Soft Instance Contrastive loss** (2025): separa clases bajo desbalance (bate a
  focal loss). L = BCE + λ·SoftIC.
- **NMS sobre la señal densa** para sacar el pico único (esto sí lo hacíamos).
- **Bi-GRU > transformers** para secuencias largas de este tipo (medido).

## Conclusiones para el rumbo (a decidir)

El diagnóstico es que **nos faltan tres cosas del SotA**, por orden de impacto probable:

1. **AUDIO** (Hallazgo 3): la palanca más grande y específica de pádel. F1 92 % con
   audio+pose vs nuestro ~70 %. El golpe SUENA. Barato de añadir (el vídeo trae audio).
2. **Contexto temporal largo + arquitectura PES** (Hallazgo 2): Bi-GRU sobre
   ventanas largas en vez de TCN corto; posiblemente sobre RGB, no solo pose.
3. **Loss adecuada a PES** (Hallazgo 4): etiqueta gaussiana + context-aware/SoftIC
   en vez de BCE plano.

**Opción de fondo:** replantear como PES sobre RGB (E2E-Spot) — abandonar el
pipeline pose+pelota para la DETECCIÓN y usar píxeles directamente, que es lo que
hace el SotA. La pose+pelota se reservaría para la CLASIFICACIÓN del tipo.

Fuentes:
- Survey PES: arXiv 2505.03991 (Action Spotting and Precise Event Detection in Sports)
- E2E-Spot: arXiv 2207.10213, repo github.com/jhong93/spot
- PES con desbalance: arXiv 2503.00147
- Pádel audio+pose F1 92 %: Decorte et al., CVPRW 2024 (Multi-Modal Hit Detection
  in Padel), IEEE 10677985
- T-DEED: Xarles et al., CVPR 2024

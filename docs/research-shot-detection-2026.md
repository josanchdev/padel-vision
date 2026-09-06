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

## Nota técnica — descarga de audio/vídeo de YouTube (feb 2026)

YouTube endureció su anti-bot: descargar audio/vídeo con yt-dlp da HTTP 403 y
"SABR-only streaming". **Fix que funcionó** (documentado para no repetir la pelea):
1. Instalar **deno** (JS runtime) — sin unzip: bajar el .zip del release y
   descomprimir con `python -c "import zipfile..."`.
2. Instalar el provider de PO token: `uv tool install yt-dlp --with
   bgutil-ytdlp-pot-provider`, y clonar su server (`Brainicism/bgutil-ytdlp-pot-provider`),
   `cd server && deno install` para las deps del `generate_once.ts`.
3. **Actualizar yt-dlp** a >= 2026.08.19 (mejor soporte SABR); versiones viejas
   fallan aunque el PO token esté.
4. Descargar con `yt-dlp --js-runtimes deno -f "..."` (el provider genera el
   PO token con deno automáticamente).

Aprendizaje de datos: el vídeo 1080p60 original se perdió (crash WSL / limpieza);
el audio del vídeo completo no alineaba con las etiquetas hechas sobre el 1080p.
Solución: re-descargar el 1080p CON audio fusionado (mismo archivo) para que
vídeo+audio+etiquetas estén sincronizados.

## Prueba de concepto del AUDIO (feb 2026) — señal útil pero no trivial

Con el fix de descarga se obtuvo el audio de citys_cup y se analizó el onset:
- **El pop del golpe SÍ es detectable en juego limpio**: en rallies sin jaleo los
  golpes aparecen como picos agudos, aislados y separados (evidencia
  `docs/media/audio_highpass_3khz.png`, golpes en s303-305). Un filtro paso-alto
  3 kHz (el pop es agudo, la voz grave) los realza.
- **PERO con aplausos/público el audio se satura** (muro de energía HF en s308-312
  del mismo gráfico) y el golpe no se distingue de un aplauso con detección de
  picos simple.
- **Normalización global engañosa**: normalizar el onset por el máximo de 65 min
  (un aplauso fuerte) aplasta los golpes normales a ~0,02-0,05.

**Lección:** el audio-solo con picos de energía NO basta (recall bajo con jaleo).
Coincide con el paper de pádel: ellos usan un MODELO APRENDIDO sobre el audio
(aprende la forma del pop) COMBINADO con pose (audio+pose → F1 92 %), no picos
crudos. El audio es una señal fuerte a FUSIONAR, no un detector por umbral.

**Obstáculo de datos:** los vídeos con GT (PadelTracker100) NO tienen audio; el
audio de YouTube no alinea con las etiquetas de citys_cup (el vídeo 1080p original
del etiquetado se perdió, YouTube re-sirvió otra versión). Para probar audio+GT hay
que re-etiquetar unos golpes sobre el vídeo actual (con audio sincronizado) o
grabar/conseguir vídeo de pádel con audio Y etiquetas alineadas.

## Pipeline de audio EXACTO del SotA de pádel (Decorte et al., leído del PDF)

Del PDF completo del paper (Decorte et al., CVPRW 2024). NO usan detección de
picos (lo que probamos y falló) — usan un modelo **SED (Sound Event Detection)**
aprendido:

**Features:** 40 bins **log-Mel** (rango 0-42 kHz), FFT window 2048, 50% overlap,
en secuencias de 256 frames. (El espectrograma mel = "imagen" tiempo×frecuencia.)

**Arquitectura (SED-net adaptada, CRNN, 109k params):**
```
conv2D 1→64 (3x3) ReLU + maxpool(1x5)
conv2D 64→64 (3x3) ReLU + maxpool(1x2)
conv2D 64→64 (3x3) ReLU + maxpool(1x2)
reshape (12x64)→(256x3)
bidirectional GRU (256x3)→(256x32) tanh
bidirectional GRU (256x32)→(256x16) tanh
time distributed dense →(256x16)
time distributed dense →(256x1) sigmoid   # per-frame prob de golpe
```
**Loss: binary focal cross-entropy** (para el desbalance). Adam lr 0.001, batch 128.

**Ellos mismos reconocen el ruido:** el espectrograma "indica golpes pero también
tiene ruido de pasos, jugadores hablando, aplausos" → por eso el modelo APRENDE a
distinguir, no un umbral (confirma por qué nuestro detector de picos falló).
Ellos también usan **TrackNet** para la pelota (igual que nosotros) para saber qué
jugador golpea.

**DATASET ABIERTO (el GT de audio que nos faltaba):**
`https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3`
- 5 h 28 min de pádel (11 torneos), VÍDEO + AUDIO
- **2377 golpes anotados** con ventanas de inicio/fin
- 319 con "qué jugador golpeó"
Cita obligatoria: Decorte et al., "Multi-Modal Hit Detection and Positional
Analysis in Padel Competitions", CVPRW 2024 (ver docs/bibliography.md).

**Plan a decidir:** replicar el CRNN SED sobre log-Mel entrenado con ESTE dataset
(audio+GT alineado, resuelve nuestro obstáculo de datos). Es la vía con F1 92 %
demostrado en nuestro deporte exacto.

## Lectura COMPLETA del paper de audio — detalles clave (feb 2026)

Tras leer el PDF entero (no solo el resumen), detalles que enriquecen el plan:

**Asignación de jugador ("quién golpea") — receta robusta, replicable:**
NO es "muñeca más cercana en 1 frame". Es un **voto ponderado sobre ventana de
500ms (12 frames @25fps)** alrededor del golpe, con peso por distancia
euclídea estandarizada (ec. 1), y fallbacks en cascada: pose→muñecas (mín. de las
dos), sin pose→centro del bbox, sin bbox→bbox promedio en ±2s. Barrido secundario
que usa que los equipos golpean ALTERNÁNDOSE para rellenar huecos. **Resultado:
83,7% acierto jugador, 86,8% equipo.** → La pelota+pose SÍ sirve para el "quién"
con esta lógica (nuestro intento previo era ingenuo, 1 frame). Confirma el instinto
de Jorge de mantener la pelota.

**Validación (sed_eval, collar 250ms, 4 splits cross-rally 70/30):** F1 92%
(σ 1,6%), error rate 0,16 = casi todo DELETIONS (golpes perdidos), no falsos. Los
perdidos: sobre todo slices/dejadas (suenan poco, poco representadas) — confirma la
preocupación de Jorge sobre golpes suaves. Falsos positivos: pala contra marco
metálico, jugador contra cristal, pala al suelo → candidatos a filtrar con el RGB
verificador.

**Implicaciones para nuestro plan:** (1) el "quién" se replica con voto multi-frame
+ alternancia de equipos (resuelve la duda #1 de ADR-0015). (2) El audio pierde
dejadas/slices → el RGB (que las ve aunque no suenen) las recupera: refuerza el
valor del RGB como verificador/segunda señal.

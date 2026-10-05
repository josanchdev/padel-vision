# Guía para escribir la memoria

Reglas para que todos los capítulos sigan el mismo criterio. Recoge lo que se ha ido
acordando (fecha de esta versión: 5 de octubre de 2026). Si una regla cambia, se cambia
aquí primero.

Compilar: `latexmk -pdf tfg.tex` dentro de `memoria/`. Las notas `\alumno{}` y
`\tutor{}` solo se ven con `\drafttrue` (`config/config.tex`).

---

## 1. Estructura

Normativa de la ETSII (2026, art. 16): portada, resumen, palabras clave, índices,
introducción y bibliografía son obligatorios; el resto, recomendado. **Entre 40 y 60
páginas, contadas desde la introducción y sin los anexos.**

| Capítulo | Págs. | Qué va | De dónde sale |
|---|---|---|---|
| Introducción | ~5 | Contexto, el problema (cuándo, quién y qué tipo de golpe), aportaciones, estructura | `docs/sistema.md` |
| Objetivos | ~3 | Objetivo general, específicos (se retoman uno a uno en las conclusiones), alcance | `docs/sistema.md` |
| Estado del arte y alternativas | ~8 | Raqueta y pádel, instante del golpe, personas/seguimiento/pelota, acciones por esqueleto, alternativas descartadas con su medida | `notas_citas.md`, `docs/experiments.md` |
| Metodología | ~4 | Desarrollo por niveles, ADR y bitácora, metodología experimental, herramientas, planificación | ADR, `docs/experiments.md` |
| Descripción informática | ~16 | Especificación, arquitectura, datos, cada paso del sistema, visor (una frase), pruebas | `docs/sistema.md`, ADR |
| Experimentos y validación | ~12 | Protocolo y métricas, cada paso por separado, sistema completo, errores, decisiones medidas, coste, limitaciones | `docs/metrics/` |
| Conclusiones y trabajo futuro | ~3 | Objetivos frente a resultados, lecciones, trabajo futuro | `docs/sistema.md` |

**Anexos** (no cuentan páginas). Cada uno se cita desde el texto («véase el
anexo A»), y en un anexo nunca aparece un resultado que no esté en el cuerpo.

- Reproducibilidad: instalar el repo y regenerar cada cifra (`README.md`, `docs/metrics/README.md`).
- Registro de decisiones: índice de los ADR con su estado (`docs/decisions/README.md`).
- Herramientas de etiquetado: las dos herramientas propias, con capturas nuevas (apartado 5).
- Visor web: capturas. La web es una demostración, no una prueba de que el sistema generalice;
  en el cuerpo basta una frase.

El tutor validó esta estructura el 5 de octubre de 2026, sin anexo de uso de IA.

## 2. Cifras

- **Ninguna cifra se escribe a mano.** Un script genera desde `docs/metrics/` las tablas, las
  figuras y un fichero de macros con las cifras que aparecen en el texto (pendiente de hacer).
  Así no pueden desincronizarse.
- Cada resultado va con su intervalo de confianza del 95 %: «82,7 % (IC 95 %: 80,3–85,0)».
- **Decimales:** porcentajes con uno («82,7 %») y F1, macro-F1 y demás proporciones con tres
  («0,958»). Es lo habitual en visión por computador, coherente entre sí (0,958 = 95,8 %) y
  acorde con la incertidumbre: con intervalos de ±2 puntos o más, el segundo decimal no
  significa nada. Las cifras de otros trabajos se redondean igual, sin añadirles decimales que no
  dan (el 83,70 % de Decorte → 83,7 %; su F1 0,92 se queda como está).
- Los resultados de clasificadores retirados (ST-GCN, PoseConv3D) no se comparan con el
  final: otro dataset y otras clases.

**Redacciones acordadas (no cambiar sin hablarlo):**

| Tema | Se dice | No se dice |
|---|---|---|
| Asignación frente a Decorte | «comparable, por encima en la media» (la diferencia no es concluyente) | «supera» |
| Detección frente a Decorte | «reproduce» (otro criterio para dar por buena una detección, ver Mesaros) | «supera» |
| Clasificador | «basado en el método de BST (que parte de TemPose)» | «reimplementa BST»; no se enumeran las diferencias menores |
| Detector de pelota | «basado en TrackNet: una red de tipo U-Net con un refinador residual propio» | «TrackNetV3» (es un nombre interno); «una U-Net» |
| Pose | «el modelo de pose, preentrenado en COCO [n], devuelve 17 puntos por persona» | — |
| Intervalos | «intervalos del 95 % por *bootstrap* [Efron], calculados por Monte Carlo con 10 000 remuestreos de puntos» (Monte Carlo, una sola vez) | atribuir a Efron el intervalo percentil |
| Por qué audio | nuestra propia medición (pose + pelota frente a audio, mismos datos) y Decorte | — |
| Trabajo previo | «hasta donde sabemos (búsqueda de octubre de 2026), la clasificación del tipo de golpe en pádel se ha abordado con sensores de muñeca [Cartes Domínguez]; con vídeo de retransmisión no hemos encontrado trabajos revisados por pares» | — |
| Productos comerciales | «existen productos comerciales que anuncian estadísticas por tipo de golpe [aiball, Padelytics, Padel AI], pero no publican su método ni su precisión» | detalles de cada uno |
| La pulsera (Cartes Domínguez) | cómo funciona y sus limitaciones | sus resultados (no son comparables) |
| Crecimiento del pádel | «según la FIP, …» (cifras declaradas por las federaciones) | darlas como hecho medido |
| Etiquetas de golpe de PadelTracker100 | no se mencionan | — |

## 3. Citas

- **Criterio:** se cita lo que se dice, cuando es obligatorio o aporta. La memoria se escribe
  desde lo que conviene contar, no desde las referencias. `notas_citas.md` (fuera del repo,
  junto a los PDF) es la reserva: qué dice cada fuente y en qué página.
- Las claves están en `bibliografia.bib`. BibTeX solo imprime las citadas.
- Forma: `texto~\cite{clave}` antes del punto; varias en una orden:
  `\cite{a,b}` → «[1], [2]»; tres o más seguidas → «[3]–[5]». Cada número enlaza con la
  bibliografía.
- Estilo IEEE: con más de seis autores, el primero y «et al.»; ya está configurado.

## 4. Estilo y vocabulario

- Español, frases cortas, un término por concepto. Nada que no se pueda defender ante el
  tribunal.
- **Números (RAE):** coma decimal; millares sin punto: «2377 golpes», «77 300 pistas» (los de
  cuatro cifras, juntos; desde cinco, con espacio fino); símbolos separados de la cifra:
  «82,7 %», «48 kHz», «15 px». En LaTeX, con `\num{}` y `\SI{}{}` (apartado 7), que lo hacen
  solos.
- **Comillas:** «» primero, “” dentro, ‘’ en último lugar.
- **Palabras extranjeras sin adaptar, en cursiva (RAE):** *frame*, *dataset*,
  *accuracy*, *recall*, *bootstrap*… Se mantienen los términos estándar en visión por computador; el glosario dice
  cuáles.
- **Siglas:** completas la primera vez: «red neuronal convolucional (CNN)».
- Títulos de capítulos y apartados: solo la primera palabra en mayúscula.

**Glosario** (se amplía al escribir):

| Usar | No usar | Nota |
|---|---|---|
| punto | rally, peloteo | Avisar una vez de que los datasets los llaman *rallies* |
| golpe; tipo de golpe | — | Clases: derecha, revés, remate, saque |
| jugador J1–J4 | — | |
| pareja | equipo | Los dos jugadores del mismo lado |
| pelota | bola | |
| *frame* | fotograma | Estándar en visión por computador |
| seguimiento | *tracking* | |
| asignación del golpe | — | Quién golpea |
| *dataset* | conjunto de datos | Los nombres propios (CVSPORTS, PadelTracker100) tal cual, sin cursiva |
| *accuracy*, *precision*, *recall*, F1, macro-F1 | acierto, exactitud, precisión, exhaustividad | Nombres de métricas, en inglés para ser consistentes. En sentido general («no publican su precisión») se escribe en español |
| puntos clave | *keypoints* | Los 17 puntos de la pose |

## 5. Figuras: formatos, carpetas y cómo se hacen

**Carpetas** (`memoria/figuras/`), separadas por cómo se crea cada cosa:

| Carpeta | Qué contiene | Quién la escribe |
|---|---|---|
| `capturas/` | Imágenes de píxeles: pantallas (web, herramientas) y *frames* de vídeo | Un script de capturas |
| `generadas/` | Gráficas (PDF), tablas (`.tex`) y `cifras.tex`, todo desde `docs/metrics/` | El script de la memoria, que la borra y la rehace entera: no se toca a mano |
| `diagramas/` | Diagramas en TikZ (`.tex`) | A mano: son dibujos, no datos |

Las tres van al repo, para que la memoria compile sin ejecutar nada. Nombres en español, según
el contenido, en minúsculas y con guiones (`asignacion-confusion.pdf`, `web-portada.png`).
Qué figura va en qué capítulo lo dice el apartado 6, no la carpeta.

**Formatos:**

| Qué | Formato | Por qué |
|---|---|---|
| Gráficas (barras, curvas, matrices) | PDF vectorial | Nítidas a cualquier zoom, texto seleccionable |
| Diagramas | TikZ | Vectoriales y con la misma letra que el texto |
| Pantallas (web, herramientas) | PNG | Sin pérdida: el texto de la pantalla queda limpio |
| *Frames* de vídeo | JPG, calidad 95 | Es una foto: en PNG pesaría unas diez veces más sin diferencia visible |

Un PNG metido en un PDF no gana calidad: lo que cuenta es la resolución.

**Cómo se hace cada imagen.** Ninguna con recortes de pantalla; todas con un script, el código y
los modelos finales, para poder rehacerlas igual:

- **Web:** navegador sin pantalla (Playwright) a escala 2 (3840×2160): el texto se lee nítido
  aunque se amplíe.
- **Herramientas de etiquetado:** el script llama a su propio código de dibujo con un *frame*
  real y guarda la imagen tal cual: los píxeles exactos, sin bordes de ventana.
- ***Frames* del partido:** del vídeo original (no del exportado, que está recomprimido), con el
  código de dibujo del sistema.
- Nunca se escala una imagen hacia arriba: el máximo es la resolución del vídeo (1920×1080, unos
  16 cm a 300 ppp).
- **Gráficas:** al tamaño final en la página, con la letra de la memoria (Latin Modern, instalada)
  y un cuerpo parecido al del texto, para no tener que reescalarlas.

`docs/media/` no se usa en la memoria: guarda las imágenes que enlaza la bitácora.

**Reglas de uso:**

- **Título:** debajo en las figuras, encima en las tablas (estilo IEEE; ya configurado). Que se
  entienda sin leer el texto.
- Cada figura y tabla se cita en el texto antes de aparecer. Dejar que LaTeX las coloque
  (`[htbp]`); no forzar con `[H]`.

## 6. Figuras y tablas por capítulo

Plan del 5 de octubre de 2026: unas 19 figuras y 12 tablas. Se revisará al verlas generadas y al
escribir cada capítulo.

| Capítulo | Figura o tabla | Cómo se hace |
|---|---|---|
| Introducción | Fig.: *frame* de retransmisión con esqueletos, J1–J4, pelota y etiqueta del golpe | captura de un *frame* |
| Objetivos | Tabla: cada objetivo y cómo se mide | texto |
| Estado del arte | Tabla: trabajos en pádel (qué resuelve cada uno: cuándo, quién, tipo; con qué datos) | texto |
| Metodología | Fig.: planificación julio–diciembre por niveles (Gantt) | gráfica desde las fechas de la bitácora |
| Metodología | Fig.: validación por torneos (entrenar con 10, medir en el que queda fuera) | TikZ |
| Metodología | Tabla: herramientas y versiones | desde `uv.lock` |
| Descripción | Fig.: arquitectura, el flujo en 5 pasos | TikZ |
| Descripción | Tabla: requisitos funcionales y no funcionales | texto |
| Descripción | Tabla: *datasets* y para qué se usa cada uno | desde los JSON |
| Descripción | Fig.: el *dataset* propio por clase y por torneo | gráfica |
| Descripción | Fig.: cuándo. Espectrograma log-Mel de un punto, golpes reales y salida del detector frente al umbral | ejecuta el sistema sobre un punto |
| Descripción | Fig.: jugadores. Esqueleto de 17 puntos y pista con sus 6 puntos y vista cenital | ejecuta el sistema sobre un punto |
| Descripción | Fig.: pelota. Mapa de calor de la red y trayectoria antes y después de limpiarla | ejecuta el sistema sobre un punto |
| Descripción | Fig.: quién. Ventana de voto de un golpe real (distancia pelota–muñeca de los 4 jugadores) | ejecuta el sistema sobre un punto |
| Descripción | Fig.: qué tipo. El clasificador: rama de pose, rama de pelota, transformer, regla del saque | TikZ |
| Experimentos | Tabla 1: cada paso por separado, con intervalos y la columna de Decorte | desde los JSON |
| Experimentos | Tabla 2: el sistema completo por torneos | desde los JSON |
| Experimentos | Fig.: por qué umbral 0,5 (F1, *precision* y *recall* con la dispersión de 3 ejecuciones) | gráfica |
| Experimentos | Fig.: matriz de confusión de la asignación | gráfica |
| Experimentos | Fig.: asignación punto a punto frente a Decorte (16 puntos) | gráfica |
| Experimentos | Fig.: matriz de confusión del tipo de golpe | gráfica |
| Experimentos | Fig.: *accuracy* del tipo por torneo, con su intervalo | gráfica |
| Experimentos | Fig.: embudo, de cada 100 golpes cuántos se detectan, con jugador y con tipo correctos | gráfica |
| Experimentos | Fig.: errores típicos (2 o 3 *frames* reales) | captura de *frames* |
| Experimentos | Tabla: decisiones medidas (sin y con cada una) | desde los JSON |
| Experimentos | Tabla: detector de pelota a 15, 30 y 60 px, con y sin limpieza | desde los JSON |
| Experimentos | Tabla: coste por etapa | desde los JSON |
| Conclusiones | Tabla: cada objetivo frente a su resultado | texto + cifras |
| Anexos | Tablas: qué comando regenera cada cifra; los ADR y su estado. Capturas de las dos herramientas y de la web | texto y capturas |

Fuera de la memoria (siguen en el repo): las curvas de entrenamiento, la matriz de confusión del
tipo en la cadena completa (repite la del clasificador) y las figuras de exploración.

## 7. LaTeX

Configurado en `config/config.tex`: T1 + Latin Modern, `microtype`, `booktabs`, `siunitx`
(coma decimal) y `cite` (citas al estilo IEEE).

- **Tablas:** `\toprule`, `\midrule`, `\bottomrule`. Nunca líneas verticales ni dobles, ni
  `\hline`. Las unidades en la cabecera, no en cada celda. Siempre «0,958», nunca «,958».
  Columnas de números con `S[table-format=…]` para alinearlas por la coma.
- **Números:** `\num{0.958}` → 0,958; `\num{2377}` → 2377; `\SI{82.71}{\percent}` → 82,71 %;
  `\SI{48}{\kilo\hertz}` → 48 kHz. Se escribe con punto y sale con coma.
- **Referencias cruzadas:** en minúscula y sin partir de línea: `la figura~\ref{fig:pipeline}`,
  `la tabla~\ref{tab:paso-a-paso}`, `el capítulo~\ref{cap:experimentos}`. Etiquetas con
  prefijo: `cap:`, `sec:`, `fig:`, `tab:`, `eq:`.
- **Comillas y acentos:** se escriben tal cual (UTF-8): «», tildes, ñ.
- **Cursiva:** `\emph{}`.

## 8. Cómo trabajamos

1. Capítulo a capítulo. Primero acordamos qué va, en puntos, sin escribir nada.
2. Borrador del capítulo.
3. Jorge lo revisa y lo reescribe con sus palabras.
4. Compila sin avisos y se hace el commit.

Nada se escribe sin acuerdo previo. El tutor dio el visto bueno a esta forma de trabajar.

## 9. Revisión antes de entregar

- [ ] Compila sin errores, sin avisos y sin referencias ni citas rotas.
- [ ] Cada figura y tabla, citada en el texto; cada anexo, citado desde el cuerpo.
- [ ] Siglas definidas la primera vez; glosario respetado.
- [ ] Ninguna cifra a mano; ningún «2.377» ni «82,7%» sin espacio.
- [ ] Redacciones acordadas del apartado 2 respetadas.
- [ ] Entre 40 y 60 páginas desde la introducción, sin anexos.
- [ ] Portada y datos iguales que en la plataforma de TFG.
- [ ] Notas `\alumno{}` y `\tutor{}` eliminadas o desactivadas (`\draftfalse`).
- [ ] Revisión ortográfica.
- [ ] La comprobación de plagio que indique el tutor.

---

**Fuentes:**
[RAE, DPD: números](https://www.rae.es/dpd/n%C3%BAmeros) ·
[RAE, DPD: porcentajes](https://www.rae.es/dpd/porcentajes) ·
[RAE: tipos de comillas](https://www.rae.es/espanol-al-dia/cuando-se-usa-cada-tipo-de-comillas) ·
[booktabs](https://ctan.org/pkg/booktabs) ·
[siunitx](https://ctan.org/pkg/siunitx) ·
[microtype](https://ctan.org/pkg/microtype) ·
[cite](https://ctan.org/pkg/cite) ·
manual de babel-spanish (TeX Live) ·
normativa de TFG de la ETSII (2026).

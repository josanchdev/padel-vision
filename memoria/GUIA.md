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

- **Ninguna cifra se escribe a mano.** `scripts/build_thesis_assets.py` genera desde
  `docs/metrics/` las tablas, las figuras y `cifras.tex`; en el texto, `\cifra{tipo-accuracy}`
  imprime «82,7 %» (las claves están en `figuras/generadas/cifras.tex`). Así no pueden
  desincronizarse.
- Cada resultado va con su intervalo de confianza del 95 %: «82,7 % (IC 95 %: 80,3–85,0)».
- **Decimales:** porcentajes con uno («82,7 %») y F1, macro-F1 y demás proporciones con tres
  («0,958»). Es lo habitual en visión por computador, coherente entre sí (0,958 = 95,8 %) y
  acorde con la incertidumbre: con intervalos de ±2 puntos o más, el segundo decimal no
  significa nada. Las cifras de otros trabajos se redondean igual, sin añadirles decimales que no
  dan (el 83,70 % de Decorte → 83,7 %; su F1 0,92 se queda como está).
- **Excepción, las matrices de confusión:** proporciones por fila con dos decimales («0,80»),
  como las muestra por defecto scikit-learn; con tres, dieciséis celdas se leen mal.
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

### Redacción

Lo primero es la calidad. Estas reglas existen para que el texto sea propio y profesional; ninguna
justifica explicar algo peor. Coincidir con otros textos en un hecho («el pádel crece») o en la
forma estándar de explicar algo técnico (la pose, una red convolucional) no es plagio: plagio es
copiar texto o ideas sin citarlos.

**Voz y tiempos.**
- Impersonal con «se» («se ha entrenado», «se midió») o con un sujeto concreto («el sistema
  detecta», «el clasificador recibe»). Nunca «hemos» ni «nosotros»; el trabajo es de una persona.
  Para referirse a él, «este trabajo».
- Presente para describir el sistema y lo que muestran tablas y figuras; pretérito perfecto para
  lo hecho en este trabajo («se ha etiquetado»); pasado simple para trabajos ajenos («Decorte et
  al. propusieron»); futuro solo en el trabajo futuro.

**Texto propio.**
- Nunca se escribe con un artículo abierto delante: las notas de `notas_citas.md` dicen qué
  afirma cada fuente y en qué página, y la frase se escribe desde nuestros datos. Las ideas ajenas
  llevan cita aunque estén reescritas; no hay citas literales.
- Las afirmaciones generales se apoyan en un dato concreto con su fuente y, cuando aporta, en algo
  propio del trabajo (los 2377 golpes, VIGO, la cámara de retransmisión). Sin convertirlo en
  fórmula: no todos los párrafos acaban en «por eso este trabajo».
- `docs/sistema.md` y los ADR son fuente de contenido, no texto que se copia.

**Lo que delata un texto automático.**
- Muletillas vacías: «cabe destacar», «es importante señalar», «en este sentido», «juega un
  papel crucial», «en el panorama actual», «no solo… sino también», «robusto», «innovador», «sin
  precedentes», tríos de adjetivos, párrafos que acaban repitiendo lo que acaban de decir. Se usan
  solo si en esa frase concreta dicen algo.
- Formato de chat: listas con negritas al empezar cada punto en mitad del texto, exceso de rayas
  (—), preguntas como título.
- Restos de la conversación: «como se acordó», la guía, nombres de personas, notas internas.
  Ficheros y funciones del repositorio, solo cuando hacen falta (sobre todo en el anexo de
  reproducibilidad).

**Coherencia.** Cada cifra con `\cifra{}`; cada cita con su página en `notas_citas.md`; cada
afirmación sobre el código comprobada contra el código (función, parámetro, valor); el glosario,
igual en todos los capítulos. Si una frase no se sabría defender ante el tribunal, se reescribe o
se quita.

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

**Criterio** (acordado el 6 de octubre de 2026, tras revisar una primera tanda): una figura o tabla
solo entra si responde a una pregunta concreta del capítulo, se entiende en pocos segundos y no se
puede decir igual de bien en una frase. Si no, la cifra va en el texto con `\cifra{}`. Se decide
**capítulo a capítulo, al escribirlo**, viendo cómo queda junto al texto.

**Ya generadas** (`figuras/generadas/`, se pueden quitar si al escribir no aportan):

| Pieza | Para qué capítulo | Pregunta que responde |
|---|---|---|
| `tabla-pasos` | Experimentos | ¿Cuánto acierta cada paso y cómo queda frente a Decorte? |
| `tabla-sistema` | Experimentos | ¿Y todo encadenado? (se prefirió al embudo, que repetía lo mismo) |
| `tipo-confusion` | Experimentos | ¿En qué se equivoca el clasificador? (derecha y revés) |
| `umbral-detector` | Experimentos | ¿Por qué umbral 0,5? |
| `dataset-por-torneo` | Descripción (datos) | ¿Qué significa evaluar por torneos? |
| `tabla-datasets` | Descripción (datos) | ¿Qué datos se usan y para qué? |
| `tabla-coste` | Experimentos | ¿Es práctico? ¿Qué etapa cuesta más? |
| `tabla-herramientas` | Metodología | ¿Con qué está hecho? |

**Descartadas** (sus cifras van en frases): la tabla de decisiones medidas, la del detector de
pelota a varias tolerancias, la asignación punto a punto frente a Decorte, el tipo por torneo, la
matriz de confusión de la asignación y el embudo.

**Candidatas, a decidir al llegar a cada capítulo:** el *frame* anotado de la introducción; las
tablas de objetivos, estado del arte, requisitos y objetivos frente a resultados; el Gantt; los
diagramas en TikZ (arquitectura, validación por torneos, clasificador); las figuras que ejecutan
el sistema sobre un punto (espectrograma, esqueleto y pista, pelota, ventana de voto); los
errores típicos; y en los anexos, las capturas de la web y de las herramientas.

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
2. Borrador del capítulo, ya con la calidad final y las figuras y tablas que necesite.
3. Jorge lo revisa: cambia lo que quiera con sus palabras y deja lo que le convence.
4. Comprobación de cifras, citas, afirmaciones sobre el código, glosario y muletillas
   (búsqueda en los `.tex`); compila sin avisos y se hace el commit.

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
- [ ] Sin «hemos» ni «nosotros», sin muletillas vacías ni restos de la conversación (apartado 4).
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

# Guía para escribir la memoria

Estructura de la memoria, de dónde sale cada cifra y cada figura, y cómo se trabaja (versión del
6 de octubre de 2026). No hay reglas de redacción: cada capítulo se escribe como mejor quede.

Compilar: `latexmk -pdf tfg.tex` dentro de `memoria/`. Las notas `\alumno{}` y
`\tutor{}` solo se ven con `\drafttrue` (`config/config.tex`).

---

## 1. Estructura

Normativa de la ETSII (2026, art. 16): portada, resumen, palabras clave, índices,
introducción y bibliografía son obligatorios; el resto, recomendado. **Entre 40 y 60
páginas, contadas desde la introducción y sin los anexos.**

| Capítulo | Págs. | Qué va | De dónde sale |
|---|---|---|---|
| Introducción | ~5 | Se replantea desde cero (6 oct): qué contar y cómo, en debate | `docs/sistema.md` |
| Objetivos | ~3 | Objetivo general, específicos (se retoman uno a uno en las conclusiones), alcance | `docs/sistema.md` |
| Estado del arte y alternativas | ~8 | Raqueta y pádel, instante del golpe, personas/seguimiento/pelota, acciones por esqueleto, alternativas descartadas con su medida | `notas_citas.md`, `docs/experiments.md` |
| Metodología | ~4 | Desarrollo por niveles, ADR y bitácora, metodología experimental, herramientas, planificación | ADR, `docs/experiments.md` |
| Descripción informática | ~16 | Especificación, arquitectura, datos, cada paso del sistema, visor, pruebas | `docs/sistema.md`, ADR |
| Experimentos y validación | ~12 | Protocolo y métricas, cada paso por separado, sistema completo, errores, decisiones medidas, coste, limitaciones | `docs/metrics/` |
| Conclusiones y trabajo futuro | ~3 | Objetivos frente a resultados, lecciones, trabajo futuro | `docs/sistema.md` |

**Anexos** (no cuentan páginas):

- Reproducibilidad: instalar el repo y regenerar cada cifra (`README.md`, `docs/metrics/README.md`).
- Registro de decisiones: índice de los ADR con su estado (`docs/decisions/README.md`).
- Herramientas de etiquetado: las dos herramientas propias, con capturas nuevas (apartado 4).
- Visor web: capturas. La web usa los modelos de producción: es una demostración, no una prueba
  de que el sistema generalice.

El tutor validó esta estructura el 5 de octubre de 2026, sin anexo de uso de IA.

## 2. Nada inventado

- **Ninguna cifra se escribe a mano.** `scripts/build_thesis_assets.py` genera desde
  `docs/metrics/` las tablas, las figuras y `cifras.tex`; en el texto, `\cifra{tipo-accuracy}`
  imprime «82,7 %» (las claves están en `figuras/generadas/cifras.tex`, y el redondeo lo hace el
  script). Así no pueden desincronizarse.
- Cada cita tiene su página en `notas_citas.md` (fuera del repo, junto a los PDF). Lo que se
  diga del código se comprueba en el código.
- No se copia texto de ninguna fuente; las ideas ajenas llevan su cita aunque estén reescritas.

## 3. Citas

- **Criterio:** se cita lo que se dice, cuando es obligatorio o aporta. La memoria se escribe
  desde lo que conviene contar, no desde las referencias. `notas_citas.md` es la reserva: qué
  dice cada fuente y en qué página.
- Las claves están en `bibliografia.bib`. BibTeX solo imprime las citadas.
- Forma: `texto~\cite{clave}` antes del punto; varias en una orden:
  `\cite{a,b}` → «[1], [2]»; tres o más seguidas → «[3]–[5]». Cada número enlaza con la
  bibliografía.
- Estilo IEEE: con más de seis autores, el primero y «et al.»; ya está configurado.
- **Nombrar o solo numerar.** Si la frase habla del trabajo citado (qué hicieron, en qué se basa
  este), se nombra: «Decorte et al.~\cite{...} detectan…». Si la fuente solo respalda un dato, basta
  el número al final.

## 4. Figuras: formatos, carpetas y cómo se hacen

**Carpetas** (`memoria/figuras/`), separadas por cómo se crea cada cosa:

| Carpeta | Qué contiene | Quién la escribe |
|---|---|---|
| `capturas/` | Imágenes de píxeles: pantallas (web, herramientas) y *frames* de vídeo | Un script de capturas |
| `generadas/` | Gráficas (PDF), tablas (`.tex`) y `cifras.tex`, todo desde `docs/metrics/` | El script de la memoria, que la borra y la rehace entera: no se toca a mano |
| `diagramas/` | Diagramas en TikZ (`.tex`) | A mano: son dibujos, no datos |

Las tres van al repo, para que la memoria compile sin ejecutar nada. Nombres en español, según
el contenido, en minúsculas y con guiones (`asignacion-confusion.pdf`, `web-portada.png`).
Qué figura va en qué capítulo lo dice el apartado 5, no la carpeta.

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

## 5. Figuras y tablas por capítulo

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

**Candidatas, a decidir al llegar a cada capítulo:** el *frame* anotado y la captura del visor
para la introducción; las tablas de objetivos, estado del arte, requisitos y objetivos frente a
resultados; el Gantt; los diagramas en TikZ (arquitectura, validación por torneos, clasificador);
las figuras que ejecutan el sistema sobre un punto (espectrograma, esqueleto y pista, pelota,
ventana de voto); los errores típicos; y en los anexos, las capturas de la web y de las
herramientas.

## 6. LaTeX

Configurado en `config/config.tex`: T1 + Latin Modern, `microtype`, `booktabs`, `siunitx`
(coma decimal: `\num{0.958}` → 0,958; `\SI{48}{\kilo\hertz}` → 48 kHz) y `cite` (citas al
estilo IEEE). Etiquetas con prefijo: `cap:`, `sec:`, `fig:`, `tab:`, `eq:`.

## 7. Cómo trabajamos

1. **Contenido.** Capítulo a capítulo, primero se acuerda qué va, en puntos, sin escribir nada.
2. **Borrador.** Claude escribe el capítulo completo, con la mejor calidad posible y las figuras y
   tablas que necesite.
3. **Debate.** Jorge lo comenta en el propio `.tex`, en una línea propia encima de la frase:
   `% JORGE: …` (no sale en el PDF). Al decir «listo», se discute cada comentario y el borrador se
   ajusta hasta que Jorge lo valida.
4. **Versión de Jorge.** Jorge escribe su versión a partir del borrador validado, a su manera, y la
   sustituye en el `.tex`. El texto final es suyo.
5. **Revisión.** Claude revisa esa versión: que no haya nada inventado ni incorrecto y que todo
   tenga sentido y se entienda. Solo sugiere; los cambios los hace Jorge. Después compila sin
   avisos y se hace el commit.

El tutor dio el visto bueno a esta forma de trabajar.

## 8. Revisión antes de entregar

- [ ] Compila sin errores, sin avisos y sin referencias ni citas rotas.
- [ ] Cada figura y tabla, citada en el texto; cada anexo, citado desde el cuerpo.
- [ ] Ninguna cifra a mano.
- [ ] Entre 40 y 60 páginas desde la introducción, sin anexos.
- [ ] Portada y datos iguales que en la plataforma de TFG.
- [ ] Notas `\alumno{}` y `\tutor{}` eliminadas o desactivadas (`\draftfalse`); ningún
  `% JORGE` pendiente en los `.tex`.
- [ ] Revisión ortográfica.
- [ ] La comprobación de plagio que indique el tutor.

---

**Fuentes:**
[booktabs](https://ctan.org/pkg/booktabs) ·
[siunitx](https://ctan.org/pkg/siunitx) ·
[microtype](https://ctan.org/pkg/microtype) ·
[cite](https://ctan.org/pkg/cite) ·
normativa de TFG de la ETSII (2026).

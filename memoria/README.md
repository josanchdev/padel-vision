# Memoria del TFG

Fuente LaTeX de la memoria. Se compila con `latexmk` dentro de esta carpeta (configuración en `.latexmkrc`) o al guardar en VS Code con LaTeX Workshop; ambos usan pdfLaTeX y biber.

## Cómo se escribe

- Cada capítulo es un fichero de `pages/` (anexos en `pages/anexos/`); `tfg.tex` los reúne y `config/config.tex` tiene toda la configuración.
- **Una frase por línea** en los `.tex`: no cambia el PDF y en git cada cambio se ve frase a frase.
- **Citas**: `\cite{clave}`, con las claves de `bibliografia.bib`.
- **Referencias**: `\cref{fig:x}` escribe «figura 2.4» y `\Cref` lo pone en mayúscula a principio de frase. Prefijos de etiqueta: `cap:`, `sec:`, `fig:`, `tab:`, `anexo:`.
- **Siglas**: se declaran en `config/acronimos.tex` y se usan con `\ac{clave}`; la primera vez sale la forma larga y la lista se genera sola.
- **Números**: `\num{0.958}` y `\qty{82.7}{\percent}` los escriben como pide la RAE (0,958 y 82,7 %).
- **Figuras**: en `figuras/`, cada una generada por un script del repositorio, que dice de qué datos sale (por ejemplo, `scripts/build_problem_figure.py` para `figuras/problema.pdf`). No se editan a mano: se cambia el script y se regenera.
- **Comentarios de revisión**: `\tutor{}`, `\cotutor{}` y `\alumno{}`; desaparecen al poner `\draftfalse` en `config/config.tex`.
- **Corrector**: extensión LTeX+ de VS Code, configurada en español.

## Origen

Plantilla de la ETSII [codeurjc-students/tfg-template](https://github.com/codeurjc-students/tfg-template) (commit `4df8109`, licencia MIT en `LICENSE`), derivada de la plantilla oficial de la Escuela que está en el Aula Virtual. Se copió la parte LaTeX (sin `.devcontainer/`, `tex2site/`, `.github/` ni `.vscode/`, porque se compila con TeX Live nativo) y sobre ella se aplicaron las decisiones de abajo. `bibliografia.bib` es la bibliografía ya revisada del trabajo.

## Lo que exige la normativa

Reglamento Específico del TFG de la ETSII, aprobado en Junta de Escuela el 16 de enero de 2026.

- **Obligatorio** (art. 16): portada según el anexo I (tutor y cotutor), resumen de una página, palabras clave, índices, introducción y bibliografía.
- **Recomendado** (art. 16): objetivos (con la descripción del problema, el estudio de alternativas y la metodología), descripción informática (especificación, diseño, implementación y pruebas), experimentos y validación, conclusiones (logros y trabajo futuro) y apéndices.
- **Extensión** (art. 16): entre 40 y 60 páginas, contadas desde la introducción y sin apéndices.
- **Formato** (art. 16): DIN A4 y formato legible; el reglamento pone como ejemplo márgenes de 2,5 cm, Times New Roman 12 e interlineado 1,5. La plantilla LaTeX se recomienda, no se exige.
- **Licencia** (art. 8 y anexo IV): si se concede libre distribución, la licencia debe figurar en la memoria (CC BY, CC BY-ND o CC BY-SA 4.0).
- **Plagio** (anexo III): antes de autorizar la defensa, el tutor genera en la plataforma de TFG un informe de plagio de la memoria.

## Lo que evalúa el tribunal

Rúbrica del acta de calificación de la ETSII (Aula Virtual). La memoria decide casi toda la nota: el bloque escrito y la calidad del trabajo se juzgan sobre ella.

| Bloque | Peso | Indicador (peso en el bloque) | Dónde se juega |
|---|---|---|---|
| Presentación escrita | 30 % | Problema descrito y contextualizado; bibliografía que recoge todas las citas, con referencias autorizadas (25 %) | Introducción, estado del arte |
| | | Objetivos formulados de forma clara y explícita, alineados con el problema (25 %) | Objetivos |
| | | Propuesta descrita de forma clara y organizada (25 %) | Descripción informática |
| | | Reflexión sobre el cumplimiento de los objetivos, valoración global con su aplicación y relevancia, e interpretación de los resultados (25 %) | Experimentos, conclusiones |
| Calidad del trabajo (mínimo 5) | 50 % | Metodología apropiada y bien aplicada (30 %) | Metodología |
| | | Complejidad técnica alta: técnicas complejas o novedosas (70 %) | Toda la memoria |
| Presentación oral | 20 % | Claridad, terminología y tiempo (50 %); respuestas al tribunal (50 %) | Defensa |

## Estructura

Introducción; Objetivos; Estado del arte y alternativas; Metodología; Datos; Descripción informática; Experimentos y validación; Conclusiones y trabajo futuro; Bibliografía; Anexos (reproducibilidad, herramientas de etiquetado y visor web). Es la estructura validada por el tutor con un capítulo propio para los datos, que estaban dentro de Descripción informática: el conjunto de datos etiquetado es una aportación del trabajo y lo usan tanto el sistema como la evaluación. Antes de los capítulos: portada, licencia, agradecimientos, resumen con palabras clave, índice, índices de figuras y tablas, y siglas.

## Decisiones

**Formato** (elegido comparando muestras con el mismo texto):

- **Latin Modern 11 pt, márgenes de 2,5 cm, interlineado de 1,5 líneas y párrafos separados por un espacio, sin sangría.** Se lee mejor que el formato de la plantilla (12 pt y márgenes de unos 3 cm), cuyos márgenes dejan un 20 % menos de superficie de texto por página; 11 pt queda cerca del ejemplo de la normativa sin apretar el texto como 10 pt.
- **Dos caras, con cada capítulo en página impar.** La memoria se imprimirá, además de entregarse en PDF. Los márgenes son iguales en las dos caras, así que en pantalla no se nota.
- **Cabecera y pie como en una memoria impresa**: capítulo en la cabecera de las páginas pares y sección en la de las impares; número de página por fuera, autor en el pie de las impares y escuela en el de las pares. La primera página de cada capítulo lleva solo el número de página, centrado. Las páginas en blanco que quedan antes de un capítulo van vacías del todo, sin cabecera ni pie, como pide la norma tipográfica (repetir la cabecera en ellas es un efecto por defecto de LaTeX, no una convención).
- **Índices en el orden de la plantilla de la Escuela**: índice general, índice de tablas, índice de figuras y, después, siglas.
- **Títulos de capítulo grandes**, con los tamaños y posiciones medidos en la memoria de referencia: «Capítulo N» a 40 pt y el título a 35 pt; los capítulos sin número (índices, resumen, siglas, bibliografía) a 40 pt. Las secciones, al tamaño estándar (14,4 pt). La normativa no dice nada sobre los títulos; «Capítulo N» es lo que hacen las plantillas de la Escuela.
- **Índices a interlineado sencillo** y sin mini índice al principio de cada capítulo.
- **Enlaces sin color**, para que la versión impresa salga en negro; siguen siendo clicables en el PDF.
- **Portada de la plantilla** (ya sigue el anexo I) con los dos tutores; la licencia, en el reverso.

**Técnicas:**

- **pdfLaTeX con codificación T1 y `microtype`.** Sin T1, pdfLaTeX divide mal las palabras con tilde y el texto copiado del PDF sale roto; `microtype` mejora el justificado. LuaLaTeX no aporta nada que haga falta aquí y compila más lento.
- **biblatex + biber con el estilo IEEE**, en lugar de BibTeX con `IEEEtran.bst`, que escribe la bibliografía en inglés. Con biblatex sale en español; se corrigen las pocas cadenas que el estilo no traduce y la puntuación junto a las comillas (`config/config.tex`). Las citas seguidas se agrupan ([1]-[4], con guion, como pide la RAE para los intervalos).
- **Español con babel**: «Tabla» y no «Cuadro», «Índice», «Anexo» y numeración de apartados sin punto final (1.2 y no 1.2.).
- **Números con `siunitx`** según la RAE: coma decimal (como escribe la propia normativa), espacio antes de las unidades y del %, y separación de miles solo a partir de cinco cifras (2377, pero 10 000).
- **`cleveref`** para las referencias cruzadas, con «tabla» y «anexo» en lugar de «cuadro» y «capítulo».
- **`booktabs`** para las tablas (sin líneas verticales) y título encima de cada tabla.
- **`acro`** para las siglas: la primera aparición se escribe completa y la lista se genera sola.
- **El PDF compilado no se versiona** (`.gitignore`): se regenera en cada compilación. Tampoco los documentos de la universidad (`Datos_uni_Esco/`).

## Pendiente

- **Licencia** de la memoria (y del código), a acordar con el tutor; por ahora, la CC BY-SA de la plantilla.
- **Resumen en inglés**: la normativa no lo exige; se decide al escribir el resumen.
- **Revisión del tutor**: Overleaf o PDF, cuando haya borrador. Antes de pasarle los `.tex`, borrar los comentarios `% IDEAS` del principio de cada capítulo.
- **Versión final**: `\draftfalse` para quitar los comentarios de revisión.

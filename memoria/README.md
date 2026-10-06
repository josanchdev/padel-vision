# Memoria del TFG

Fuente LaTeX de la memoria. Se compila con `latexmk -pdf tfg.tex` (o al guardar en VS Code con LaTeX Workshop).

## Origen

Plantilla de la ETSII [codeurjc-students/tfg-template](https://github.com/codeurjc-students/tfg-template) (commit `4df8109`, licencia MIT en `LICENSE`), derivada de la plantilla oficial de la Escuela. Se copió tal cual salvo `bibliografia.bib`, que es la bibliografía ya revisada del trabajo, y los espacios en blanco al final de línea y de fichero, que normalizan los *hooks* de pre-commit del repositorio. No se copiaron `.devcontainer/` (entorno Docker), `tex2site/` (genera una web a partir de la memoria), `.github/` ni `.vscode/`, porque se compila con TeX Live nativo.

## Lo que exige la normativa

Reglamento Específico del TFG de la ETSII, aprobado en Junta de Escuela el 16 de enero de 2026.

- **Obligatorio** (art. 16): portada según el anexo I (tutor y cotutor), resumen de una página, palabras clave, índices, introducción y bibliografía.
- **Recomendado** (art. 16): objetivos (con la descripción del problema, el estudio de alternativas y la metodología), descripción informática (especificación, diseño, implementación y pruebas), experimentos y validación, conclusiones (logros y trabajo futuro) y apéndices.
- **Extensión** (art. 16): entre 40 y 60 páginas, contadas desde la introducción y sin apéndices.
- **Formato** (art. 16): DIN A4 y formato legible; el reglamento pone como ejemplo márgenes de 2,5 cm, Times New Roman 12 e interlineado 1,5.
- **Licencia** (art. 8 y anexo IV): si se concede libre distribución, la licencia debe figurar en la memoria (CC BY, CC BY-ND o CC BY-SA 4.0).
- **Plagio** (anexo III): antes de autorizar la defensa, el tutor genera en la plataforma de TFG un informe de plagio de la memoria.

## Estructura de partida

La validada por el tutor: Introducción; Objetivos; Estado del arte y alternativas; Metodología; Descripción informática; Experimentos y validación; Conclusiones y trabajo futuro; Bibliografía; Anexos (reproducibilidad, herramientas de etiquetado y capturas del visor). La plantilla todavía tiene su estructura de ejemplo, pensada para aplicaciones web.

## Decisiones

Se añaden aquí según se toman, cada una con su motivo.

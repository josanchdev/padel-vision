# Compilación de la memoria: pdfLaTeX y biber (latexmk lo detecta por biblatex).
# La usan igual la terminal (`latexmk`) y LaTeX Workshop en VS Code.
@default_files = ('tfg.tex');
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$clean_ext = 'bbl run.xml synctex.gz';

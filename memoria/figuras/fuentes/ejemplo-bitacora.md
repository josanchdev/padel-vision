## La regla del saque (sep 2026)

**Problema.** En la matriz de confusión del clasificador aparecían saques
predichos en golpes que no abren el punto, donde el reglamento no permite un
saque.

**Prueba.** Se comprobó sobre las etiquetas que los 97 saques son el primer
golpe de su punto, sin ninguna excepción. Por ello, se probó una regla aplicada
tras el clasificador: en los golpes que no abren el punto, el tipo se elige entre
las tres clases restantes.

**Resultados** (dejando fuera un torneo cada vez):

|                     | Sin regla | Con regla |
|---------------------|----------:|----------:|
| Precisión del saque |     0,861 |     1,000 |
| *Recall* del saque  |     0,916 |     0,916 |
| F1 del saque        |     0,888 |     0,956 |
| Macro-F1            |     0,842 |     0,859 |
| *Accuracy*          |    82,5 % |    82,7 % |

**Hallazgo.** Los 14 saques falsos estaban todos a mitad del punto, y la regla
los eliminó sin perder ningún saque real. La mejora procede del reglamento, no
del aprendizaje.

**Decisión.** Se incorporó la regla al sistema final, tras el clasificador.

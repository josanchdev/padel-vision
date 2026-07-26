# Backlog de mejoras (no bloqueantes)

Cosas que **sabemos** que mejorarían el sistema pero que hemos aparcado
conscientemente para no romper el flujo "demo siempre verde". No es una lista de
bugs ni de trabajo pendiente de un nivel abierto: es el registro de decisiones de
*posponer*, con su motivo, para que en la defensa se pueda explicar qué falta y
por qué se priorizó lo que se priorizó.

Convención: cada entrada dice **qué**, **por qué se aparcó** y **qué esperamos
ganar**. Prioridad orientativa (no es un compromiso de orden).

---

## Datos (la palanca de mayor impacto)

- **[Alta] Entrenamiento largo del detector de pelota.** La prueba corta (4000
  train / 3000 val, 6 epochs) dio TrackNetV2 F1 0,896 y V3 0,911. Falta extraer
  los ~46k frames de cada partido y entrenar más epochs para el modelo
  definitivo. *Por qué aparcado:* validar la lógica de botes/trayectoria no
  necesita el modelo perfecto, y un job largo arriesga un crash de WSL; se lanza
  cuando se pueda vigilar. *Ganancia esperada:* F1 más alto y estable, error de
  localización menor.
- **[Alta] Más datos propios (vídeos de YouTube / cámara URJC).** Buscar
  partidos de pádel en YouTube a distintas alturas/calidades y grabar con la
  cámara del equipo URJC. *Por qué aparcado:* PadelTracker100 basta como
  kickstarter para tener modelos funcionando; ampliar datos es mejora, no
  requisito. *Ganancia esperada:* generalización real (ADR-0005) y dataset
  propio que diferencia el TFG ([[feedback-no-dataset-dependency]]).
- **[Media] Qué hacer al tener esos datos nuevos.** Flujo previsto: (1) anotar
  en CVAT (pista + pelota + tipo de golpe + oclusión), (2) reentrenar detector
  de pelota, clasificador de golpes y, si aporta, el detector de pista, (3)
  medir mejora contra el split actual, (4) documentar en `experiments.md`. Es el
  mismo patrón "kickstarter → v1 → iterar" del dataset de vehículos de noche en
  Ferrovial ([[user-jorge-profile]]).
- **[Media] Anotar oclusión de pelota nosotros.** El flag `occluded` de
  PadelTracker100 está vacío (0/19.386), así que no se puede sacar el desglose
  visible-vs-ocluida que mostraría dónde gana V3. *Ganancia esperada:*
  desbloquea esa gráfica comparativa (la infraestructura ya está lista).

## Modelos

- **[Media] Ampliar la taxonomía de golpes a 6 clases.** El clasificador usa 5
  (la dejada se absorbe en "otro", solo 25 ejemplos, ADR-0008). Con datos
  propios etiquetados, la dejada puede ser clase propia.
- **[Baja] Barrido de hiperparámetros de la pelota.** `pos_weight`, `sigma` del
  heatmap, tolerancia de la métrica, resolución del grid. Ahora usan valores
  razonables por defecto; MLflow ya está listo para trazar el barrido.
- **[Baja] Explorar sucesores de PoseConv3D / TrackNetV3.** Papers como punto de
  partida, no techo ([[feedback-modern-tools]]); revisar si hay algo de 2026 que
  mejore de forma clara antes de invertir en entrenarlo.

## Pipeline / producto

- **[Alta] Conectar el clasificador de golpes entrenado a la API/web.** El
  PoseConv3D está entrenado pero aún no expuesto en el producto; se hará en la
  sesión de rediseño web.
- **[Media] Rediseño web con estándares UX 2026.** Plataforma de suscripción
  profesional: mapas de calor, resúmenes, stats por jugador. Sesión propia con
  investigación de UX y decisión de paleta (rojo URJC vs verde pádel).
- **[Baja] Latencia del clasificador de golpes.** Emite ~0,5 s tras el impacto
  (ventana centrada necesita frames posteriores). Aceptable en batch; revisar si
  se quiere modo live (Nivel 3 del roadmap).

## Infra / robustez

- **[Media] Diagnosticar los crashes de WSL.** Mitigados (RAM 20 GB, commits
  frecuentes, entrenamientos reanudables) pero sin causa raíz. *Ganancia:* poder
  lanzar jobs largos sin vigilancia.

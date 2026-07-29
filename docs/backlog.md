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

## Robustez del vídeo real: la pila de dependencias hacia el análisis de punto

Un vídeo de pádel NO es rally continuo: entre puntos los jugadores caminan,
recogen pelotas, hablan; hay repeticiones a cámara lenta, planos del público,
primeros planos de un jugador. Contar golpes/botes sobre TODO el vídeo
indiscriminadamente contamina cualquier estadística (gestos de calentamiento
contados como golpes, pelota muerta rodando contada como bote, detector de
pelota y homografía enloquecidos en el plano del público). Antes de explotar los
datos hay que **generar datos fiables**. Esta es la pila, de abajo (resolver
primero) a arriba (bonus final):

1. **[Alta] ¿La cámara está en la pista?** Distinguir frames de pista de
   público / repetición / primer plano. Sin esto, todo lo demás es ruido. Es el
   primer eslabón y el más prioritario.
2. **[Alta] ¿Hay un punto en juego?** Segmentación temporal del vídeo en
   "puntos" vs "no-puntos" (jugadores caminando, calentando, recogiendo). Define
   la unidad de análisis. **Motivación UI extra (Jorge, 29 jul):** la cronología
   por TIEMPO no escala a partidos largos (20 min = cientos de marcas ilegibles).
   La solución es navegar por PUNTOS (como los replays de un juego: "Punto 1,
   Punto 15" → saltas a ese punto, con sus stats). Eso requiere esta
   segmentación. Así que la segmentación de puntos no es solo para stats de
   resultado — también para que la cronología/navegación tenga sentido. La
   cronología por tiempo actual sirve para clips cortos; la vista por puntos es
   el objetivo para partidos completos.
3. **[Media] Límites de cada punto:** saque → último golpe válido. La unidad
   sobre la que se calcula cualquier resultado.
4. **[Media] Secuencia ordenada de golpes+botes por punto** (quién, cuándo, qué
   golpe). Ya la empezamos a tener con Nivel 2 (golpes) + Nivel 3 (botes); falta
   ensamblarla por punto una vez existan los límites del punto (3).
5. **[Baja/exploración] ¿Cómo terminó el punto?** Winner (nadie llegó) vs error
   (a la red / fuera / doble bote). Requiere reglas de pádel + geometría. Aquí
   empieza lo genuinamente difícil.
6. **[Bonus final, exploración no comprometida] Atribución del resultado y stats
   por jugador.** Qué equipo suma, qué golpe fue decisivo, quién falló y con qué
   ("J4 falló el 60% de reveses"), mapas de calor, resúmenes. Es la CIMA de la
   pila: agregación estadística sobre el nivel 5.

**Decisión de alcance (para la memoria/defensa):** los niveles 1-4 son
alcanzables y sólidos. Los niveles 5-6 son un salto grande, en terreno donde
incluso productos comerciales fallan y con vídeo monocular hay ambigüedades a
veces irresolubles (¿doble bote o el jugador llegó justo?). Se tratan como
**trabajo futuro / exploración de hasta dónde llegamos con los datos**, NO como
funcionalidad prometida. Honesto ante el tribunal: reconoce el reto abierto en
vez de comprometer algo frágil. Coherente con "modelos/datos perfectos primero,
funcionalidades encima" ([[project-padel-vision-tfg]]).

## Modelos

- **[EN CURSO — ADR-0013] Detector de golpe aprendido (Modelo 1).** Frente grande
  del TFG. Sustituye la heurística ADR-0012 (recall 15-52%). Plan de arranque para
  la próxima sesión de código (partes CPU-only, no bloquean el entreno de la
  pelota):
  1. **Dataset builder** (CPU): variante de `padel_cv/clip_builder.py`. Reusa
     `assemble_clips_from_match` (ya recorta ventanas de 32f centradas en golpe +
     muestrea NoShot), pero (a) etiqueta BINARIO golpe/no-golpe en vez de tipo, y
     (b) AÑADE el canal de trayectoria de pelota a cada ventana (de `*_ball.json`
     GT para entrenar; de nuestro TrackNet en inferencia). GT: los 906 bloques
     `has_shot` de PadelTracker100 (ver `padel_ml/shot_eval.load_shot_blocks`).
  2. **Red temporal** (CPU): arquitectura ligera pose+pelota (estilo TCN+transformer
     de TemPose/BST leído en el código oficial; ver experiments.md). Empezar
     simple: TCN(pose)+TCN(pelota) → temporal → cabeza binaria.
  3. **Entrenar + medir** (GPU): recall/precision con `padel_ml.shot_eval` (mismo
     harness), objetivo ~85-95%. Comparar heurística vs aprendido.
- **[Baseline listo] Clasificador pose-solo honesto** en `runs/shot_baseline/`
  (val0 + val1, ~0,60 macro-F1, sin el data leakage del checkpoint archivado).
  Es la referencia a batir con pose+pelota (Modelo 2, ADR-0013).
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

- **[HECHO 28 jul] Conectar el clasificador de golpes entrenado a la API/web.**
  El worker pasa `shot_model` (config `PADEL_SHOT_MODEL`) → usa el PoseConv3D
  real en vez del dummy. Docker instala `packages/ml`; los pesos llegan por el
  volumen `./runs/archive`. Pelota aún NO conectada (se reentrena en Fase 1;
  se cablea al tener el modelo bueno).
- **[Media] Rediseño web con estándares UX 2026.** Plataforma de suscripción
  profesional. Sesión propia con investigación de UX y decisión de paleta (rojo
  URJC vs verde pádel). Ojo: las stats por jugador que exhibiría dependen de la
  pila de robustez de arriba (niveles 5-6); la web puede empezar por lo que ya
  es fiable (trayectoria, golpes, botes, minimapa) y sumar stats cuando existan.
- **[Media] Marcar el golpe exacto dentro del clip del modal.** En pádel 6 s
  (±3 s) pueden contener 2-4 golpes, así que el clip del modal muestra varios y
  no se distingue CUÁL es el golpe que se pulsó. Idea (Jorge, 27 jul 2026):
  señalarlo visualmente. Opciones a valorar: (a) un marcador en la barra de
  progreso en el instante del golpe; (b) resaltar solo al jugador que golpea en
  ese frame; (c) reducir el margen (p. ej. ±1,5 s); (d) auto-pausar en el frame
  del golpe. Datos ya disponibles (`frame_index`/`timestamp_s` del golpe).
- **[Baja] Latencia del clasificador de golpes.** Emite ~0,5 s tras el impacto
  (ventana centrada necesita frames posteriores). Aceptable en batch; revisar si
  se quiere modo live (Nivel 3 del roadmap).

## Infra / robustez

- **[Media] Diagnosticar los crashes de WSL.** Mitigados (RAM 20 GB, commits
  frecuentes, entrenamientos reanudables) pero sin causa raíz. *Ganancia:* poder
  lanzar jobs largos sin vigilancia.

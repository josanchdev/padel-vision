# ADR-0014: Etiquetado de golpes por marca rápida (frame + tipo)

**Estado:** aceptada · 2026-07-31

## Contexto

El sistema de golpes está limitado por los DATOS, no por la arquitectura, medido
tres veces (ADR-0013 y bitácora): heurística 15-52% recall, detector aprendido
0,60 en test balanceado pero **29% recall / 24% precisión al deslizar sobre vídeo
real**. Con 906 golpes de 2 partidos no hay atajo de ingeniería. Hace falta
etiquetar más (Fase B), pero el etiquetado "completo" estilo PadelTracker100
(mover keypoints de pose en ~10 frames por golpe) es demasiado caro (~2-3
min/golpe) para llegar a miles.

**Idea de Jorge (coincide con el SotA ShuttleSet/BST):** no anotar keypoints —
ver el vídeo, pausar en cada golpe y marcar solo `(frame, tipo)`. El sistema
recorta la ventana ±N frames con la pose+pelota que YA detectamos.

## Decisiones

- **A — Una marca, dos modelos.** Cada golpe se anota con UNA marca temporal +
  tipo. El frame entrena el DETECTOR (dónde hay golpe; los frames sin marca son
  negativos) y el tipo entrena el CLASIFICADOR. Una sola pasada de etiquetado
  alimenta ambos modelos (Modelo 1 y Modelo 2 de ADR-0013). No hay dos tareas.
- **B — Refinado por pelota.** El anotador pausa cerca del impacto (importa para
  el detector); el sistema afina la marca al frame donde la pelota está más cerca
  de la muñeca del golpeador. Marca aproximada del humano → frame exacto por
  geometría. En ese frame, si la pelota está mal detectada, se corrige a mano
  (solo ese frame, no el vídeo entero).
- **C — Taxonomía = gesto (4 clases):** Saque, Derecha, Revés, Remate.
  **Descartados:** volea (→ derecha/revés según gesto), bandeja (ambigua/rara),
  dejada/chiquita (se confunde con volea suave, pocos ejemplos, rompería métricas
  como "Other" a 0,18). **Globo/lob descartado (31 jul, Jorge):** no es un gesto
  distinto — es una derecha/revés con trayectoria alta, INDISTINGUIBLE en el frame
  de contacto (la ventana que ve el modelo); solo se sabe viendo la pelota subir
  DESPUÉS. Jorge lo detectó al anotar (tenía que rectificar constantemente). Un
  globo se etiqueta por su gesto (derecha/revés). Si el dato "es globo" resulta
  valioso, se añade como FLAG (como la pared), no como tipo. El eje del gesto
  queda limpio y todo distinguible en el contacto.
- **D — Golpe de PARED = flag aparte, no un tipo.** Una salida de pared se
  ejecuta con derecha o revés; "pared" es de dónde vino la pelota (contexto), no
  el gesto. Se anota como columna `de_pared` (sí/no), metadato analítico que no
  ensucia la clasificación de gesto. CSV: `frame, tipo, de_pared, jugador`.
- **E — Herramienta a medida, no CVAT.** CVAT es para mover cajas/keypoints
  (lento). Este flujo es marca-y-clasifica: un reproductor OpenCV ligero con
  teclas rápidas (ESPACIO pausa, ←/→ frame, saltos, 1-5 tipo, tecla flag pared,
  deshacer, guardar), con pose+pelota dibujadas para ver errores. ~5-10 seg/golpe
  vs 2-3 min. Objetivo: 2-3 partidos continuos HD → ~1000-1500 golpes (de 906 a
  ~2000+). Empezar por partido CONTINUO (citys_cup); best_points cambia de pista
  → reservado para TEST de generalización.

## Consecuencias

- (+) Etiquetado ~15-30× más rápido que anotar keypoints → miles de golpes
  viables a mano.
- (+) Una sola pasada entrena detector Y clasificador; el CSV es simple y lo
  consume el builder existente (`shot_detect_dataset`) con mínimos cambios.
- (+) Gesto limpio + pared como metadato = mejor señal para el modelo y dato
  analítico extra sin coste de clasificación.
- (−) La calidad del detector depende de la puntería temporal del anotador (el
  refinado por pelota mitiga, pero un desfase grande no se corrige solo).
- (−) Herramienta propia que mantener (pequeña; OpenCV). No reusa CVAT, que
  seguirá para pista/pelota si hiciera falta anotación fina.
- (−) La pose sigue viniendo de nuestro YOLO (no se revisa a mano en masa); si la
  pose falla en un golpe, esa ventana entra ruidosa. Aceptable a 1080p (pose
  buena); el anotador puede saltar golpes con pose obviamente rota.

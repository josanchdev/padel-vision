# ADR-0007: Cola de trabajos — Arq sobre Redis

**Estado:** aceptada · 2026-07-17

## Contexto

Procesar un partido tarda varios minutos de GPU: hacerlo dentro de la petición HTTP daría timeout. Hace falta desacoplar la subida (respuesta inmediata con un id) del procesado (worker en segundo plano), con estado consultable. Opciones: Arq, Celery, RQ, o un `BackgroundTask` de FastAPI sin cola.

## Decisión

**Arq** (async, Redis) como cola de trabajos. Descartado `BackgroundTask`: no sobrevive a reinicios ni da progreso real, y habría que reescribirlo. Descartado Celery: maquinaria pesada y síncrona pensada para flotas; para un solo nodo con GPU es sobreingeniería. Arq es async nativo (encaja con FastAPI), mínimo en piezas y código, mismo autor que Pydantic/FastAPI.

Todo el stack es software libre y autohospedado (coste cero para el TFG): Arq (MIT), Redis (AGPLv3; alternativa Valkey/BSD si se requiere), FastAPI (MIT). Coste solo aparecería con hosting público 24/7, fuera del alcance académico.

## Consecuencias

- (+) API async coherente de arriba a abajo; worker desacoplado que reusa el pipeline de `packages/cv` sin cambios.
- (+) Estado de job (pending/processing/done/failed) y progreso persistidos en Redis, consultables por el front.
- (−) Redis pasa a ser dependencia de infraestructura (contenedor en Docker Compose).
- (−) Un solo worker GPU procesa en serie; paralelizar requeriría más GPUs (irrelevante a la escala del TFG).

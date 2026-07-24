#!/usr/bin/env bash
# Local development stack: Redis in Docker, API and GPU worker native.
#   ./tools/dev.sh up     -> start redis, api (:8000) and worker
#   ./tools/dev.sh down   -> stop api, worker and redis
#   ./tools/dev.sh logs   -> tail api and worker logs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/storage/logs"
PID_DIR="$ROOT/storage/pids"
COURT_MODEL="${PADEL_COURT_MODEL:-$ROOT/runs/archive/court_broadcast_v6.pt}"

start() {
    mkdir -p "$LOG_DIR" "$PID_DIR"
    docker compose -f "$ROOT/docker-compose.yml" up -d redis
    export PADEL_COURT_MODEL="$COURT_MODEL"
    export PADEL_STORAGE_DIR="${PADEL_STORAGE_DIR:-$ROOT/storage}"

    uv run --directory "$ROOT" uvicorn padel_api.main:app --host 0.0.0.0 --port 8000 \
        >"$LOG_DIR/api.log" 2>&1 &
    echo $! >"$PID_DIR/api.pid"

    uv run --directory "$ROOT" arq padel_api.worker.WorkerSettings \
        >"$LOG_DIR/worker.log" 2>&1 &
    echo $! >"$PID_DIR/worker.pid"

    echo "API      -> http://localhost:8000/docs"
    echo "Modelo   -> $COURT_MODEL"
    echo "Logs     -> $LOG_DIR/{api,worker}.log"
}

stop() {
    for name in api worker; do
        if [ -f "$PID_DIR/$name.pid" ]; then
            kill "$(cat "$PID_DIR/$name.pid")" 2>/dev/null || true
            rm -f "$PID_DIR/$name.pid"
        fi
    done
    docker compose -f "$ROOT/docker-compose.yml" down
    echo "stack parado (los datos de Redis se conservan)"
}

case "${1:-}" in
    up) start ;;
    down) stop ;;
    logs) tail -f "$LOG_DIR/api.log" "$LOG_DIR/worker.log" ;;
    *) echo "Uso: ./tools/dev.sh {up|down|logs}"; exit 1 ;;
esac

#!/usr/bin/env bash
# CVAT wrapper: run the annotation tool in Docker without touching the project.
#   ./cvat.sh up      -> start CVAT at http://localhost:8080 (data persists in volumes)
#   ./cvat.sh down    -> stop containers (annotations are KEPT in Docker volumes)
#   ./cvat.sh status  -> show running CVAT containers
#   ./cvat.sh user    -> create the admin user (first time only)
set -euo pipefail

CVAT_VERSION="v2.69.0"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVAT_DIR="$DIR/cvat-src"
export CVAT_HOST="${CVAT_HOST:-localhost}"
export CVAT_VERSION

ensure_source() {
    if [ ! -d "$CVAT_DIR" ]; then
        echo "Cloning CVAT $CVAT_VERSION (first run only)..."
        git clone --depth 1 --branch "$CVAT_VERSION" https://github.com/cvat-ai/cvat.git "$CVAT_DIR"
    fi
}

case "${1:-}" in
    up)
        ensure_source
        docker compose -f "$CVAT_DIR/docker-compose.yml" up -d
        echo
        echo "CVAT arrancando en http://localhost:8080 (tarda ~1 min el primer arranque)."
        echo "Primera vez: ejecuta './cvat.sh user' para crear tu usuario."
        ;;
    down)
        docker compose -f "$CVAT_DIR/docker-compose.yml" down
        echo "CVAT parado. Las anotaciones se conservan en los volúmenes de Docker."
        ;;
    status)
        docker ps --filter "name=cvat" --format "table {{.Names}}\t{{.Status}}"
        ;;
    user)
        docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
        ;;
    *)
        echo "Uso: ./cvat.sh {up|down|status|user}"
        exit 1
        ;;
esac

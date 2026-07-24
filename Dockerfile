# Single image for both the API and the GPU worker (they differ only in the
# command). Torch's cu13 wheels bundle the CUDA runtime, so a plain Python base
# plus the host NVIDIA driver (via the container toolkit) is enough for the GPU.
FROM python:3.11-slim-bookworm

# OpenCV needs libGL/glib; ffmpeg is provided by imageio-ffmpeg, not the system.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependency layer first (cached until manifests change).
COPY pyproject.toml uv.lock ./
COPY packages/cv/pyproject.toml packages/cv/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
RUN uv sync --all-packages --frozen --no-install-workspace

# Workspace source.
COPY packages ./packages
RUN uv sync --all-packages --frozen

# Bake the pose model so the container is self-contained (no runtime download).
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n-pose.pt')"

EXPOSE 8000
CMD ["uvicorn", "padel_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

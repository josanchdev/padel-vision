"""Minimal API shell. Real endpoints arrive with the Level 1 batch-processing flow."""

from fastapi import FastAPI

app = FastAPI(title="Padel Vision API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

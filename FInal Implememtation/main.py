from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import run_inference


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_PATH = ROOT_DIR / "frontend" / "index.html"
STATIC_DIR = ROOT_DIR / "static"

app = FastAPI(title="Brain Tumour Detection")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    image_data: str


def _decode_image_data(image_data: str) -> bytes:
    try:
        _, encoded = image_data.split(",", 1)
    except ValueError:
        encoded = image_data

    try:
        return base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image payload.") from exc


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: PredictionRequest) -> dict[str, object]:
    image_bytes = _decode_image_data(payload.image_data)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    try:
        return run_inference(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please verify the model files and runtime.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
502
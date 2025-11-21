"""
Minimal FastAPI service that exposes the leather classifier as a JSON API.
Run locally with: uvicorn simple_api:app --host 0.0.0.0 --port 8000
"""
import io
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

CLASS_LABELS = os.getenv(
    "CLASS_LABELS",
    "Buffalo,Cow,Goat,Sheep",
).split(",")
MODEL_PATH = os.getenv("MODEL_INCEPTION_PATH", "inception.h5")
TARGET_SIZE = (299, 299)

app = FastAPI(title="Leather Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MODEL_INCEPTION_PATH not found at '{MODEL_PATH}'. Set the env var to the .h5 file."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _preprocess_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            aspect_ratio = img.width / img.height
            if aspect_ratio > 1:
                new_width = TARGET_SIZE[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = TARGET_SIZE[1]
                new_width = int(new_height * aspect_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
            paste_xy = ((TARGET_SIZE[0] - new_width) // 2, (TARGET_SIZE[1] - new_height) // 2)
            canvas.paste(img, paste_xy)

        array = np.expand_dims(np.array(canvas), axis=0)
        return preprocess_input(array)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@app.get("/health", summary="Health check")
def health() -> dict[str, Literal["ok"]]:
    return {"status": "ok"}


@app.post("/predict", summary="Classify uploaded leather image")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    model = _load_model()
    processed = _preprocess_image(raw)
    predictions = model.predict(processed)
    scores = predictions[0]
    class_index = int(np.argmax(scores))

    try:
        label = CLASS_LABELS[class_index]
    except IndexError:
        raise HTTPException(
            status_code=500,
            detail="CLASS_LABELS does not match the model output dimension",
        )

    return {
        "prediction": label,
        "confidence": float(np.max(scores)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("simple_api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

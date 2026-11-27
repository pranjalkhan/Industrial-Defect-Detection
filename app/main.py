"""
FastAPI Application for Industrial Defect Detection.

Endpoints:
- GET  /             Health check
- GET  /categories   List trained product categories
- GET  /info         Model information
- POST /predict      Upload image for defect detection (?category=bottle)
"""

import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
    title="Industrial Defect Detection API",
    description="AI-powered multi-category quality inspection system for manufacturing defect detection",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@app.get("/")
def home():
    from src.inference import list_available_categories

    return {
        "service": "Industrial Defect Detection API",
        "version": "1.1.0",
        "status": "running",
        "categories": list_available_categories(),
        "endpoints": {
            "predict": "POST /predict?category=bottle - Upload an image for inspection",
            "categories": "GET /categories - List available trained categories",
            "info": "GET /info - Model information",
            "docs": "GET /docs - Interactive API documentation",
        },
    }


@app.get("/categories")
def categories():
    """Return available trained product categories."""
    from src.inference import list_available_categories

    return {"categories": list_available_categories()}


@app.get("/info")
def model_info():
    """Return model and system information."""
    import torch
    from src.inference import list_available_categories

    cats = list_available_categories()
    return {
        "model": "ConvAutoencoder",
        "available_categories": cats,
        "model_loaded": len(cats) > 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Query("bottle", description="Product category (e.g., bottle, screw, metal_nut, cable, capsule)"),
    model_type: str = Query("patchcore", description="Model architecture: patchcore (recommended), resnet, or autoencoder"),
):
    """Upload a product image for defect detection."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG and PNG are supported.",
        )

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_SIZE // (1024*1024)} MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        from src.inference import predict_image

        result = predict_image(image, category=category, model_type=model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

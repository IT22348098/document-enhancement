from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import numpy as np
import cv2
from typing import List

from enhance import ModelEnhancer

app = FastAPI(title="DocEnhance AI", version="1.0.0")

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
enhancer = ModelEnhancer("model/best_model.keras")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": enhancer.is_loaded,
        "model_params": enhancer.param_count,
    }


@app.post("/api/enhance")
async def enhance_single(file: UploadFile = File(...)):
    """Enhance a single document image"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        contents = await file.read()
        img = read_image_from_bytes(contents)

        start = time.time()
        enhanced, psnr_val, ssim_val = enhancer.enhance(img)
        elapsed = time.time() - start

        enhanced_b64 = image_to_base64(enhanced)
        original_b64 = image_to_base64(img)

        return {
            "filename": file.filename,
            "original_image": original_b64,
            "enhanced_image": enhanced_b64,
            "psnr": round(psnr_val, 2) if psnr_val is not None else None,
            "ssim": round(ssim_val, 4) if ssim_val is not None else None,
            "processing_time": round(elapsed, 2),
            "original_size": list(img.shape[:2]),
            "status": "success",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Enhancement failed: {str(e)}")


@app.post("/api/enhance/batch")
async def enhance_batch(files: List[UploadFile] = File(...)):
    """Enhance multiple document images"""
    results = []
    for file in files:
        try:
            contents = await file.read()
            img = read_image_from_bytes(contents)

            start = time.time()
            enhanced, psnr_val, ssim_val = enhancer.enhance(img)
            elapsed = time.time() - start

            enhanced_b64 = image_to_base64(enhanced)
            original_b64 = image_to_base64(img)

            results.append(
                {
                    "filename": file.filename,
                    "original_image": original_b64,
                    "enhanced_image": enhanced_b64,
                    "psnr": round(psnr_val, 2) if psnr_val is not None else None,
                    "ssim": round(ssim_val, 4) if ssim_val is not None else None,
                    "processing_time": round(elapsed, 2),
                    "status": "success",
                }
            )
        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {"results": results, "total": len(results)}


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a grayscale NumPy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def image_to_base64(img: np.ndarray) -> str:
    """Encode a NumPy image array as a base64 PNG string."""
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

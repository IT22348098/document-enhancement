"""
Image Enhancement router — IT22348098
======================================
Handles all /api/enhance endpoints for the U-Net-based document
deblurring/denoising feature.  Other teammates' routers should live
in separate files inside this same routers/ package.
"""

import time
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from enhance import ModelEnhancer
from utils import read_image_from_bytes, image_to_base64

router = APIRouter(prefix="/api", tags=["enhancement"])

# Module-level model instance — loaded once when the router is imported.
enhancer = ModelEnhancer("model/best_model.keras")


@router.get("/health")
async def health():
    """Return backend status and model-load information."""
    return {
        "status": "ok",
        "model_loaded": enhancer.is_loaded,
        "model_params": enhancer.param_count,
    }


@router.post("/enhance")
async def enhance_single(file: UploadFile = File(...)):
    """Enhance a single document image and return base64-encoded result."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        contents = await file.read()
        img = read_image_from_bytes(contents)

        start = time.time()
        enhanced, psnr_val, ssim_val = enhancer.enhance(img)
        elapsed = time.time() - start

        return {
            "filename": file.filename,
            "original_image": image_to_base64(img),
            "enhanced_image": image_to_base64(enhanced),
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


@router.post("/enhance/batch")
async def enhance_batch(files: List[UploadFile] = File(...)):
    """Enhance multiple document images in one request."""
    results = []
    for file in files:
        try:
            contents = await file.read()
            img = read_image_from_bytes(contents)

            start = time.time()
            enhanced, psnr_val, ssim_val = enhancer.enhance(img)
            elapsed = time.time() - start

            results.append(
                {
                    "filename": file.filename,
                    "original_image": image_to_base64(img),
                    "enhanced_image": image_to_base64(enhanced),
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

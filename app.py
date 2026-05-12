"""
FastAPI server — Marathon Route Segmentation Web App
"""

import base64
import io
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from src.config import Config
from src.gpx_conversion.gpx_converter import convert_pixel_path_to_gpx
from src.marathon_route_extraction.component_filter import remove_small_components
from src.marathon_route_extraction.model import load_model, predict_mask
from src.marathon_route_extraction.path_extractor import extract_ordered_path, zhang_suen_thinning

# ── FastAPI App Setup ─────────────────────────────────────────────────────────

app = FastAPI(title="Marathon Route Extraction")

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Model singleton ───────────────────────────────────────────────────────────

_model: torch.nn.Module | None = None
_device: torch.device | None = None


def get_model() -> torch.nn.Module:
    global _model, _device
    if _model is None:
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found: {Config.MODEL_PATH}")
        _device = Config.DEVICE
        _model = load_model(str(Config.MODEL_PATH), _device)
        print(f"[info] Model loaded on device={_device}")
    return _model


def get_device() -> torch.device:
    if _device is None:
        get_model()
    return _device


# ── Utility functions ─────────────────────────────────────────────────────────

def _b64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode_mask(mask_b64: str) -> np.ndarray:
    """Decode base64 mask image to numpy array."""
    _, data = mask_b64.split(",", 1)
    buf = io.BytesIO(base64.b64decode(data))
    img = Image.open(buf).convert("L")
    return np.asarray(img, dtype=np.uint8)


# ── Request/Response Models ───────────────────────────────────────────────────

class PostprocessRequest(BaseModel):
    mask_b64: str
    min_component_area: int = 500


class PointsRequest(BaseModel):
    skeleton_b64: str | None = None
    start: list[float]  # [x, y]
    end: list[float]    # [x, y]


class ConvertGPXRequest(BaseModel):
    start: list[int]
    end: list[int]
    path: list[list[int]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(static_dir / "index.html", media_type="text/html")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and run mask prediction.
    
    Returns:
        - input_img_b64: resized input image
        - mask_b64: predicted mask
    """
    try:
        # Read uploaded file
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get model and device
        model = get_model()
        device = get_device()
        
        # Predict mask
        input_img, mask_pil = predict_mask(
            model=model,
            image_pil=image_pil,
            device=device,
            image_size=Config.IMAGE_SIZE,
            threshold=Config.THRESHOLD,
            min_component_area=20,
            opening_iterations=0,
            closing_iterations=0,
        )
        
        return JSONResponse({
            "status": "success",
            "input_img_b64": _b64(input_img),
            "mask_b64": _b64(mask_pil),
        })
    except Exception as e:
        print(f"[error] predict: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/postprocess")
async def postprocess(req: PostprocessRequest):
    """
    Perform post-processing: noise removal + skeletonization.
    
    Input:
        - mask_b64: base64-encoded mask image
        - min_area: minimum component area (default 500)
    
    Returns:
        - skeleton_b64: skeletonized mask image
    """
    try:
        # Decode mask
        mask_arr = _decode_mask(req.mask_b64)
        
        # Apply component filtering from component_filter.py
        binary = mask_arr > 127
        cleaned = remove_small_components(binary, min_area=req.min_component_area)
        
        # Skeletonization
        skeleton = zhang_suen_thinning(cleaned)
        skeleton_arr = (skeleton * 255).astype(np.uint8)
        skeleton_img = Image.fromarray(skeleton_arr, mode="L")
        
        return JSONResponse({
            "status": "success",
            "skeleton_b64": _b64(skeleton_img),
        })
    except Exception as e:
        print(f"[error] postprocess: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract_path")
async def extract_path(req: PointsRequest):
    """
    Extract ordered path from start to end on skeleton.
    
    Input:
        - skeleton_b64: base64-encoded skeleton image
        - start: [x, y] start point
        - end: [x, y] end point
    
    Returns:
        - path: list of [x, y] tuples ordered start → end
    """
    try:
        if len(req.start) != 2 or len(req.end) != 2:
            raise ValueError("start/end must be [x, y]")
        if not req.skeleton_b64:
            raise ValueError("skeleton_b64 is required")

        skeleton_arr = _decode_mask(req.skeleton_b64)
        ordered = extract_ordered_path(
            skeleton_arr,
            (int(req.start[0]), int(req.start[1])),
            (int(req.end[0]), int(req.end[1])),
        )
        if ordered is None:
            return JSONResponse({
                "status": "failed",
                "path": [],
                "message": "No connected path found between start and end.",
            })

        return JSONResponse({
            "status": "success",
            "path": [[int(x), int(y)] for x, y in ordered],
        })
    except Exception as e:
        print(f"[error] extract_path: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/convert_gpx")
async def convert_gpx(req: ConvertGPXRequest):
    """
    Convert pixel path to GPX format.
    
    Input:
        - start: [x, y]
        - end: [x, y]
        - path: [[x, y], ...]
    
    Returns:
        - gpx_content: GPX file content as string
    """
    try:
        gpx_content = convert_pixel_path_to_gpx(req.start, req.end, req.path)
        
        return JSONResponse({
            "status": "success",
            "gpx_content": gpx_content,
        })
    except Exception as e:
        print(f"[error] convert_gpx: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")

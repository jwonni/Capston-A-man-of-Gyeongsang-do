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
from src.marathon_route_extraction.model import load_model, predict_mask
from src.marathon_route_extraction.path_extractor import extract_ordered_path
from src.marathon_route_extraction.postprocess import postprocess_mask

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
    area_thresh: int = 250
    circ_thresh: float = 0.5
    skel_thresh: int = 400
    max_distance: float = 150.0
    min_fragment_size: int = 0
    line_thickness: int = 2
    morph_close_size: int = 10
    final_size_thresh: int = 0
    spur_length: int = 20


class PointsRequest(BaseModel):
    skeleton_b64: str | None = None
    start: list[float]  # [x, y]
    end: list[float]    # [x, y]


class ConvertGPXRequest(BaseModel):
    start: list[int]
    end: list[int]
    path: list[list[int]]
    homography_params: dict | None = None  # /api/georeference 응답값 전달 시 실제 지리좌표 GPX 생성


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
    Perform post-processing using the 4-step marathon path pipeline, then skeletonize.

    Steps:
        1. Main path selection (largest connected component)
        2. Shape-based noise filtering (area × circularity × skeleton_length)
        3. Fragment connection (iterative endpoint-based merging)
        4. Residual fragment removal

    Returns:
        - skeleton_b64: skeletonized result image
    """
    try:
        mask_arr = _decode_mask(req.mask_b64)

        skeleton_arr = postprocess_mask(
            mask_arr,
            area_thresh=req.area_thresh,
            circ_thresh=req.circ_thresh,
            skel_thresh=req.skel_thresh,
            max_distance=req.max_distance,
            min_fragment_size=req.min_fragment_size,
            line_thickness=req.line_thickness,
            morph_close_size=req.morph_close_size,
            final_size_thresh=req.final_size_thresh,
            spur_length=req.spur_length,
        )
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
        - homography_params: (optional) /api/georeference 응답의 homography_params.
          제공 시 실제 위도/경도 GPX를 생성하고, 없으면 픽셀 좌표를 그대로 사용한다.

    Returns:
        - gpx_content: GPX file content as string
    """
    try:
        pixel_to_geo = None
        if req.homography_params:
            from src.georeferencing.homography import HomographyTransform
            tf = HomographyTransform.from_dict(req.homography_params)
            pixel_to_geo = tf.pixel_to_geo

        gpx_content = convert_pixel_path_to_gpx(
            req.start, req.end, req.path, pixel_to_geo=pixel_to_geo
        )

        return JSONResponse({
            "status": "success",
            "gpx_content": gpx_content,
        })
    except Exception as e:
        print(f"[error] convert_gpx: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/georeference")
async def georeference(file: UploadFile = File(...)):
    """
    마라톤 이미지에서 텍스트 OCR → 카카오 API → 픽셀↔지리좌표 변환 행렬 계산.

    환경변수 KAKAO_API_KEY 또는 Config.KAKAO_API_KEY 설정이 필요하다.

    Returns:
        - anchors: 앵커 목록 (텍스트, 픽셀 좌표, 위도/경도, 재투영 오차)
        - homography_params: /api/convert_gpx 에 전달할 변환 행렬 파라미터
        - mean_reprojection_error_px: 평균 재투영 오차 (픽셀)
    """
    import os
    import tempfile

    try:
        from src.georeferencing.text_detector import run_hisam
        from src.georeferencing.ocr import run_ocr
        from src.georeferencing.anchor_builder import build_raw_anchors
        from src.georeferencing.homography import (
            HomographyTransform, mad_outlier_removal, iterative_outlier_removal,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"지리좌표 모듈 로드 실패 (Hi-SAM / Qwen 설치 필요): {exc}",
        )

    if not Config.KAKAO_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="환경변수 KAKAO_API_KEY가 설정되지 않았습니다.",
        )

    # Check Hi-SAM checkpoint exists
    if not Config.HISAM_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Hi-SAM 체크포인트를 찾을 수 없습니다: {Config.HISAM_CHECKPOINT}\n"
                "docs/HISAM_CHECKPOINT_DOWNLOAD.md 에서 다운로드 방법을 확인하세요."
            ),
        )

    contents = await file.read()
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        word_regions, _mask = run_hisam(tmp_path)
        ocr_results = run_ocr(tmp_path, word_regions)
        raw_anchors = build_raw_anchors(ocr_results, Config.KAKAO_API_KEY)

        if len(raw_anchors) < 4:
            raise HTTPException(
                status_code=422,
                detail=f"유효 앵커 부족: {len(raw_anchors)}개 (최소 4개 필요)",
            )

        anchors = mad_outlier_removal(raw_anchors, thresh=Config.KAKAO_MAD_THRESH)
        anchors = iterative_outlier_removal(
            anchors,
            max_error_px=Config.HOMOGRAPHY_MAX_ERROR_PX,
            min_anchors=Config.HOMOGRAPHY_MIN_ANCHORS,
        )

        tf = HomographyTransform(anchors)
        errors = tf.reprojection_errors(anchors)
        mean_err = sum(errors) / len(errors)

        return JSONResponse({
            "status": "success",
            "num_anchors": len(anchors),
            "mean_reprojection_error_px": round(mean_err, 2),
            "anchors": [
                {
                    "text": a[4],
                    "place": a[5],
                    "pixel_x": int(a[0]),
                    "pixel_y": int(a[1]),
                    "lat": a[2],
                    "lng": a[3],
                    "reprojection_error_px": round(e, 2),
                }
                for a, e in zip(anchors, errors)
            ],
            "homography_params": tf.to_dict(),
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] georeference: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")

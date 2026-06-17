"""
FastAPI server — Marathon Route Segmentation Web App
"""

import json
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.environ["FLAGS_use_mkldnn"] = "0"       # PaddlePaddle Windows oneDNN 버그 회피
os.environ["FLAGS_enable_pir_api"] = "0"   # PIR 런타임 속성 변환 오류 회피

import base64
import io
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel

from src.config import Config
from src.config import (
    AREA_THRESH,
    CIRC_THRESH,
    SKEL_THRESH,
    MAX_DISTANCE,
    MIN_FRAGMENT_SIZE,
    LINE_THICKNESS,
    MORPH_CLOSE_SIZE,
    FINAL_SIZE_THRESH,
    SPUR_LENGTH,
    SKEL_MORPH_CLOSE,
    MIN_DIST,
)
from src.gpx_conversion.gpx_converter import convert_pixel_path_to_gpx, fix_white_line_path
if Config.MODEL_TYPE == "segformer_unet_b2":
    from src.marathon_route_extraction.segformer_unet_b2 import load_model, predict_mask
else:
    from src.marathon_route_extraction.unet import load_model, predict_mask
from src.marathon_route_extraction.path_extractor import extract_ordered_path, auto_extract_ordered_path
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


# ── DEBUG ──────────────────────────────────────────────────────────────────────
def _debug_path_overlay(
    skeleton_arr: np.ndarray,
    path: list,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    bg_img: Image.Image | None = None,
) -> str:
    """
    Overlay the extracted pixel path on the marathon image (or skeleton as fallback).

    bg_img must be the same resolution as skeleton_arr (both 512×512 from the
    predict pipeline) so that path coordinates map 1-to-1 without any scaling.

    red squares (3×3) = extracted BFS path
    blue circle       = start point
    green circle      = end point
    """
    h, w = skeleton_arr.shape

    if bg_img is not None:
        # Convert to grayscale first to make the path more visible
        gray = bg_img.convert("L")
        base = gray.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        rgb = np.asarray(base, dtype=np.uint8).copy()
    else:
        gray = (skeleton_arr // 4).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw path as 3×3 red squares so individual pixels are visible on the map.
    for x, y in (path or []):
        ix, iy = int(x), int(y)
        if 0 <= iy < h and 0 <= ix < w:
            draw.rectangle([ix - 1, iy - 1, ix + 1, iy + 1], fill=(255, 50, 50))

    r = 7
    sx, sy = int(start_xy[0]), int(start_xy[1])
    ex, ey = int(end_xy[0]), int(end_xy[1])
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(60, 120, 255))   # blue = start
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(50, 220, 60))    # green = end
    return _b64(img)
# ── END DEBUG ──────────────────────────────────────────────────────────────────


# ── Request/Response Models ───────────────────────────────────────────────────

class PostprocessRequest(BaseModel):
    mask_b64: str
    # Use centralized defaults from src.config (module-level imports)
    area_thresh: int = AREA_THRESH
    circ_thresh: float = CIRC_THRESH
    skel_thresh: int = SKEL_THRESH
    max_distance: float = MAX_DISTANCE
    min_fragment_size: int = MIN_FRAGMENT_SIZE
    line_thickness: int = LINE_THICKNESS
    morph_close_size: int = MORPH_CLOSE_SIZE
    final_size_thresh: int = FINAL_SIZE_THRESH
    spur_length: int = SPUR_LENGTH
    skel_morph_close: int = SKEL_MORPH_CLOSE


class PointsRequest(BaseModel):
    skeleton_b64: str | None = None
    start: list[float]   # [x, y]
    end: list[float]     # [x, y]
    min_dist: float = MIN_DIST
    input_img_b64: str | None = None  # ── DEBUG: 512×512 resized marathon image for overlay


class AutoExtractRequest(BaseModel):
    skeleton_b64: str
    min_dist: float = MIN_DIST
    input_img_b64: str | None = None


class ConvertGPXRequest(BaseModel):
    start: list[int]
    end: list[int]
    path: list[list[int]]
    homography_params: dict | None = None  # /api/georeference 응답값 전달 시 실제 지리좌표 GPX 생성
    category_pixels: dict | None = None   # /api/run_ocr 응답의 category_pixels (방향 보정 + 왕복 판단)


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
        _, mask_pil = predict_mask(
            model=model,
            image_pil=image_pil,
            device=device,
            image_size=Config.IMAGE_SIZE,
            threshold=Config.THRESHOLD,
            min_component_area=20,
            opening_iterations=0,
            closing_iterations=0,
        )
        
        # input_img is now the original-resolution image; the browser already
        # holds it in state.uploadedImage, so we only return the mask.
        return JSONResponse({
            "status": "success",
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

        # postprocess_mask now returns a tuple of intermediate results
        # (main_mask, noise_mask, filtered_mask, connected_mask,
        #  final_mask, skeleton_mask, features, noise_labels, connect_log)
        res = postprocess_mask(
            mask_arr,
            area_thresh=req.area_thresh,
            circ_thresh=req.circ_thresh,
            skel_thresh=req.skel_thresh,
            max_distance=req.max_distance,
            min_fragment_size=req.min_fragment_size,
            line_thickness=req.line_thickness,
            morph_close_size=req.morph_close_size if req.morph_close_size > 0 else 0,
            final_size_thresh=req.final_size_thresh,
            spur_length=req.spur_length,
            skel_morph_close=req.skel_morph_close,
        )
        # Extract skeleton mask from returned tuple (6th element)
        if isinstance(res, tuple) or isinstance(res, list):
            skeleton_arr = res[5]
        else:
            skeleton_arr = res
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
        start_xy = (int(req.start[0]), int(req.start[1]))
        end_xy   = (int(req.end[0]),   int(req.end[1]))
        ordered = extract_ordered_path(
            skeleton_arr, start_xy, end_xy,
            min_dist=req.min_dist,
        )

        # ── DEBUG ──────────────────────────────────────────────────────────────
        # Decode the 512×512 marathon image so the overlay shares the same
        # coordinate space as the skeleton and no scaling is needed.
        bg_img: Image.Image | None = None
        if req.input_img_b64:
            _, _data = req.input_img_b64.split(",", 1)
            bg_img = Image.open(io.BytesIO(base64.b64decode(_data))).convert("RGB")
        # ── END DEBUG ──────────────────────────────────────────────────────────
        if ordered is None:
            # ── DEBUG ──────────────────────────────────────────────────────────
            debug_b64 = _debug_path_overlay(skeleton_arr, [], start_xy, end_xy, bg_img)
            # ── END DEBUG ──────────────────────────────────────────────────────
            return JSONResponse({
                "status": "failed",
                "path": [],
                "message": "No connected path found between start and end.",
                "debug_overlay_b64": debug_b64,  # ── DEBUG ──
            })
        # ── DEBUG ──────────────────────────────────────────────────────────────
        debug_b64 = _debug_path_overlay(skeleton_arr, ordered, start_xy, end_xy, bg_img)
        # ── END DEBUG ──────────────────────────────────────────────────────────
        return JSONResponse({
            "status": "success",
            "path": [[int(x), int(y)] for x, y in ordered],
            "debug_overlay_b64": debug_b64,  # ── DEBUG ──
        })
    except Exception as e:
        print(f"[error] extract_path: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto_extract_path")
async def auto_extract_path_endpoint(req: AutoExtractRequest):
    """
    Automatically determine start/end from skeleton topology and extract ordered path.

    Direction heuristic: vertical span ≥ horizontal span → bottom-to-top,
    otherwise left-to-right. Returns the same JSON format as /api/extract_path.
    """
    try:
        skeleton_arr = _decode_mask(req.skeleton_b64)
        result = auto_extract_ordered_path(skeleton_arr, min_dist=req.min_dist)
        if result is None:
            return JSONResponse({
                "status": "failed",
                "path": [],
                "message": "스켈레톤에서 경로를 찾을 수 없습니다.",
            })

        bg_img: Image.Image | None = None
        if req.input_img_b64:
            _, _data = req.input_img_b64.split(",", 1)
            bg_img = Image.open(io.BytesIO(base64.b64decode(_data))).convert("RGB")

        overlay_b64 = _debug_path_overlay(
            skeleton_arr,
            result["path"],
            tuple(result["start"]),
            tuple(result["end"]),
            bg_img,
        )

        return JSONResponse({
            "status": "success",
            "start": result["start"],
            "end":   result["end"],
            "path":  [[int(x), int(y)] for x, y in result["path"]],
            "overlay_b64": overlay_b64,
        })
    except Exception as e:
        print(f"[error] auto_extract_path: {e}\n{traceback.format_exc()}")
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
        start = req.start
        end   = req.end
        path  = req.path

        # OCR 카테고리가 있으면 방향 보정 + 왕복 판단 적용
        if req.category_pixels:
            white_line = {"start": start, "end": end, "path": path}
            fixed = fix_white_line_path(white_line, req.category_pixels)
            start = fixed["start"]
            end   = fixed["end"]
            path  = fixed["path"]

        pixel_to_geo = None
        if req.homography_params:
            from src.georeferencing.homography import HomographyTransform
            tf = HomographyTransform.from_dict(req.homography_params)
            pixel_to_geo = tf.pixel_to_geo

        gpx_content = convert_pixel_path_to_gpx(
            start, end, path, pixel_to_geo=pixel_to_geo
        )

        return JSONResponse({
            "status": "success",
            "gpx_content": gpx_content,
        })
    except Exception as e:
        print(f"[error] convert_gpx: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class ElevationsRequest(BaseModel):
    latlngs: list[list[float]]  # [[lat, lng], ...]


@app.post("/api/elevations")
async def get_elevations(req: ElevationsRequest):
    """Open-Elevation API로 위경도 목록의 고도를 조회한다."""
    try:
        from src.georeferencing.ocr import fetch_elevations
        pairs = [(float(p[0]), float(p[1])) for p in req.latlngs]
        elevations = fetch_elevations(pairs)
        if elevations is None:
            raise HTTPException(status_code=502, detail="고도 데이터를 가져올 수 없습니다.")
        return JSONResponse({"status": "success", "elevations": elevations})
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] get_elevations: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/georeference")
async def georeference(file: UploadFile = File(...)):
    """
    마라톤 이미지에서 텍스트 OCR → 카카오 API → 오프셋 캘리브레이션 → 픽셀↔지리좌표 변환 행렬 계산.

    파이프라인:
      1. Hi-SAM → polygon JSON
      2. PaddleOCR (크롭별 base/retry 두 해상도)
      3. confidence 필터 + X 병합 후처리
      4. 앵커 후보 선별 → 카카오 API 병렬 검색 → MAD 필터 (build_anchors)
      5. 1차 재투영 오차 정제 (iterative_outlier_removal)
      6. Hough Circle 마커 검출 → 점진적 오프셋 캘리브레이션
      7. 오프셋 적용 → 2차 재투영 오차 정제 → 최종 호모그래피

    환경변수 KAKAO_API_KEY 설정이 필요하다.

    Returns:
        - anchors: 앵커 목록 (텍스트, 픽셀 좌표, 위도/경도, 재투영 오차)
        - homography_params: /api/convert_gpx 에 전달할 변환 행렬 파라미터
        - mean_reprojection_error_px: 평균 재투영 오차 (픽셀)
        - applied_offset: [dx, dy] 마커 오프셋 보정값
    """
    import os
    import shutil
    import tempfile

    import cv2

    try:
        from src.georeferencing.text_detector import run_hisam_to_json
        from src.georeferencing.ocr import run_ocr
        from src.georeferencing.anchor_builder import build_anchors, is_good_anchor
        from src.georeferencing.homography import (
            HomographyTransform, iterative_outlier_removal,
            progressive_offset_calibration,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"지리좌표 모듈 로드 실패 (Hi-SAM / PaddleOCR 설치 필요): {exc}",
        )

    api_key = os.environ.get("KAKAO_API_KEY", Config.KAKAO_API_KEY)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="환경변수 KAKAO_API_KEY가 설정되지 않았습니다.",
        )

    if not Config.HISAM_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Hi-SAM 체크포인트를 찾을 수 없습니다: {Config.HISAM_CHECKPOINT}\n"
                "docs/HISAM_CHECKPOINT_DOWNLOAD.md 에서 다운로드 방법을 확인하세요."
            ),
        )

    contents = await file.read()
    suffix   = Path(file.filename).suffix if file.filename else ".jpg"
    tmp_path = None
    crop_dir = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        crop_dir = tempfile.mkdtemp(prefix="paddle_crops_")

        # Stage 1: Hi-SAM → polygon JSON
        logger = logging.getLogger("georeference")
        logger.info("[Stage1] Hi-SAM 추론 시작")
        hisam_payload = run_hisam_to_json(tmp_path)
        logger.info("[Stage1] Hi-SAM 완료: polygon %d개", hisam_payload.get("num_words", 0))

        # Stage 2: PaddleOCR (크롭별 base/retry)
        logger.info("[Stage2] PaddleOCR 시작")
        ocr_results = run_ocr(hisam_payload, crop_dir)
        logger.info("[Stage2] OCR 완료: %d개 텍스트", len(ocr_results))

        # Stage 3: 앵커 후보 선별 → 카카오 병렬 검색 → MAD 필터
        candidates = [r for r in ocr_results if is_good_anchor(r.get("text", ""))]
        logger.info("[Stage3] is_good_anchor 통과: %d개 / OCR %d개", len(candidates), len(ocr_results))
        for c in candidates:
            logger.info("[Stage3]   후보: text=%r  conf=%.3f", c.get("text"), float(c.get("confidence", 0) or 0))
        if len(candidates) < 4:
            raise HTTPException(
                status_code=422,
                detail=f"앵커 후보 부족: {len(candidates)}개 (최소 4개 필요)",
            )
        anchors_after_mad = build_anchors(candidates, api_key)

        # Stage 4: 1차 재투영 오차 정제
        anchors_pass1 = iterative_outlier_removal(
            anchors_after_mad,
            max_error_px=Config.HOMOGRAPHY_MAX_ERROR_PX,
            min_anchors=Config.HOMOGRAPHY_MIN_ANCHORS,
        )

        # Stage 5: Hough Circle 마커 오프셋 캘리브레이션
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is not None:
            (offset_dx, offset_dy), _ = progressive_offset_calibration(
                anchors_pass1, img_bgr,
                consistency_px=Config.CALIB_CONSISTENCY_PX,
                min_samples=Config.CALIB_MIN_SAMPLES,
                visualize=False,
            )
        else:
            offset_dx, offset_dy = 0.0, 0.0

        # Stage 6: 오프셋을 MAD 통과 전체에 적용 → 2차 재투영 오차 정제
        anchors_corrected = [
            (a[0] + offset_dx, a[1] + offset_dy, a[2], a[3], a[4], a[5])
            for a in anchors_after_mad
        ]
        anchors_final = iterative_outlier_removal(
            anchors_corrected,
            max_error_px=Config.HOMOGRAPHY_MAX_ERROR_PX,
            min_anchors=Config.HOMOGRAPHY_MIN_ANCHORS,
        )

        tf       = HomographyTransform(anchors_final)
        errors   = tf.reprojection_errors(anchors_final)
        mean_err = sum(errors) / len(errors)

        # 앵커 오버레이 이미지 생성 (숫자 원 + 오차 텍스트)
        anchor_overlay_b64 = None
        try:
            overlay_img = Image.open(tmp_path).convert("RGB")
            overlay_drw = ImageDraw.Draw(overlay_img)
            for idx, (a, e) in enumerate(zip(anchors_final, errors), start=1):
                px, py = int(a[0]), int(a[1])
                r = 10
                overlay_drw.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=(0, 100, 220), outline=(255, 255, 255), width=2,
                )
                num_str = str(idx)
                overlay_drw.text((px - 4, py - 7), num_str, fill=(255, 255, 255))
                overlay_drw.text((px + r + 4, py - 7), f"+/-{e:.1f}px", fill=(220, 50, 50))
            buf = io.BytesIO()
            overlay_img.save(buf, format="JPEG", quality=82)
            anchor_overlay_b64 = (
                "data:image/jpeg;base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
        except Exception as ov_err:
            logger.warning("[georeference] anchor overlay 생성 실패: %s", ov_err)

        return JSONResponse({
            "status":                     "success",
            "num_anchors":                len(anchors_final),
            "mean_reprojection_error_px": round(mean_err, 2),
            "applied_offset":             [round(offset_dx, 3), round(offset_dy, 3)],
            "ocr_results":                ocr_results,
            "anchor_overlay_b64":         anchor_overlay_b64,
            "anchors": [
                {
                    "text":                  a[4],
                    "place":                 a[5],
                    "pixel_x":               int(a[0]),
                    "pixel_y":               int(a[1]),
                    "lat":                   a[2],
                    "lng":                   a[3],
                    "reprojection_error_px": round(e, 2),
                }
                for a, e in zip(anchors_final, errors)
            ],
            "homography_params": tf.to_dict(),
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] georeference: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if crop_dir and os.path.exists(crop_dir):
            shutil.rmtree(crop_dir, ignore_errors=True)


@app.post("/api/run_ocr")
async def run_ocr_endpoint(file: UploadFile = File(...)):
    """
    Stage 5 전용: Hi-SAM + PaddleOCR 실행.

    Returns:
        - num_texts: 인식된 텍스트 수
        - ocr_results: [{text, x, y, confidence, num_segments}, ...]
    """
    import shutil
    import tempfile

    try:
        from src.georeferencing.text_detector import run_hisam_to_json
        from src.georeferencing.ocr import run_ocr, classify_ocr_to_categories
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"지리좌표 모듈 로드 실패 (Hi-SAM / PaddleOCR 설치 필요): {exc}",
        )

    if not Config.HISAM_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Hi-SAM 체크포인트 없음: {Config.HISAM_CHECKPOINT}",
        )

    contents = await file.read()
    suffix   = Path(file.filename).suffix if file.filename else ".jpg"
    tmp_path = crop_dir = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        crop_dir = tempfile.mkdtemp(prefix="paddle_crops_")

        logger = logging.getLogger("run_ocr")
        logger.info("[run_ocr] Hi-SAM 추론 시작")
        hisam_payload = run_hisam_to_json(tmp_path)
        logger.info("[run_ocr] Hi-SAM 완료: polygon %d개", hisam_payload.get("num_words", 0))

        ocr_results = run_ocr(hisam_payload, crop_dir)
        logger.info("[run_ocr] OCR 완료: %d개 텍스트", len(ocr_results))

        category_pixels = classify_ocr_to_categories(ocr_results)
        logger.info("[run_ocr] 카테고리 분류: start_finish=%d  turning_point=%d",
                    len(category_pixels["start_finish"]),
                    len(category_pixels["turning_point"]))

        return JSONResponse({
            "status":          "success",
            "num_texts":       len(ocr_results),
            "ocr_results":     ocr_results,
            "category_pixels": category_pixels,
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] run_ocr: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if crop_dir and os.path.exists(crop_dir):
            shutil.rmtree(crop_dir, ignore_errors=True)


@app.post("/api/run_georeference")
async def run_georeference_endpoint(
    file: UploadFile = File(...),
    ocr_results_json: str = Form(...),
):
    """
    Stage 6 전용: OCR 결과를 받아 카카오 API + 호모그래피 계산.

    Input (multipart/form-data):
        - file: 원본 이미지 (마커 검출용)
        - ocr_results_json: /api/run_ocr 응답의 ocr_results를 JSON 직렬화한 문자열

    Returns:
        - anchors, homography_params, anchor_overlay_b64, mean_reprojection_error_px
    """
    import shutil
    import tempfile

    import cv2

    try:
        ocr_results = json.loads(ocr_results_json)
    except Exception:
        raise HTTPException(status_code=400, detail="ocr_results_json 파싱 실패")

    try:
        from src.georeferencing.anchor_builder import build_anchors, is_good_anchor
        from src.georeferencing.homography import (
            HomographyTransform,
            iterative_outlier_removal,
            progressive_offset_calibration,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"지리좌표 모듈 로드 실패: {exc}",
        )

    api_key = os.environ.get("KAKAO_API_KEY", Config.KAKAO_API_KEY)
    if not api_key:
        raise HTTPException(status_code=400, detail="환경변수 KAKAO_API_KEY가 설정되지 않았습니다.")

    contents = await file.read()
    suffix   = Path(file.filename).suffix if file.filename else ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger = logging.getLogger("run_georeference")

        # 앵커 후보 선별 → 카카오 병렬 검색 → MAD 필터
        candidates = [r for r in ocr_results if is_good_anchor(r.get("text", ""))]
        logger.info("[run_geo] 앵커 후보: %d개 / OCR %d개", len(candidates), len(ocr_results))
        for c in candidates:
            logger.info("[run_geo]   후보: text=%r  conf=%.3f", c.get("text"), float(c.get("confidence", 0) or 0))
        if len(candidates) < 4:
            raise HTTPException(
                status_code=422,
                detail=f"앵커 후보 부족: {len(candidates)}개 (최소 4개 필요)",
            )
        anchors_after_mad = build_anchors(candidates, api_key)

        # 1차 재투영 오차 정제
        anchors_pass1 = iterative_outlier_removal(
            anchors_after_mad,
            max_error_px=Config.HOMOGRAPHY_MAX_ERROR_PX,
            min_anchors=Config.HOMOGRAPHY_MIN_ANCHORS,
        )

        # Hough Circle 마커 오프셋 캘리브레이션
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is not None:
            (offset_dx, offset_dy), _ = progressive_offset_calibration(
                anchors_pass1, img_bgr,
                consistency_px=Config.CALIB_CONSISTENCY_PX,
                min_samples=Config.CALIB_MIN_SAMPLES,
                visualize=False,
            )
        else:
            offset_dx, offset_dy = 0.0, 0.0

        # 오프셋 적용 → 2차 재투영 오차 정제 → 최종 호모그래피
        anchors_corrected = [
            (a[0] + offset_dx, a[1] + offset_dy, a[2], a[3], a[4], a[5])
            for a in anchors_after_mad
        ]
        anchors_final = iterative_outlier_removal(
            anchors_corrected,
            max_error_px=Config.HOMOGRAPHY_MAX_ERROR_PX,
            min_anchors=Config.HOMOGRAPHY_MIN_ANCHORS,
        )

        tf       = HomographyTransform(anchors_final)
        errors   = tf.reprojection_errors(anchors_final)
        mean_err = sum(errors) / len(errors)

        # 앵커 오버레이 이미지 생성
        anchor_overlay_b64 = None
        try:
            overlay_img = Image.open(tmp_path).convert("RGB")
            overlay_drw = ImageDraw.Draw(overlay_img)
            for idx, (a, e) in enumerate(zip(anchors_final, errors), start=1):
                px, py = int(a[0]), int(a[1])
                r = 10
                overlay_drw.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=(0, 100, 220), outline=(255, 255, 255), width=2,
                )
                overlay_drw.text((px - 4, py - 7), str(idx), fill=(255, 255, 255))
                overlay_drw.text((px + r + 4, py - 7), f"+/-{e:.1f}px", fill=(220, 50, 50))
            buf = io.BytesIO()
            overlay_img.save(buf, format="JPEG", quality=82)
            anchor_overlay_b64 = (
                "data:image/jpeg;base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
        except Exception as ov_err:
            logger.warning("anchor overlay 생성 실패: %s", ov_err)

        return JSONResponse({
            "status":                     "success",
            "num_anchors":                len(anchors_final),
            "mean_reprojection_error_px": round(mean_err, 2),
            "applied_offset":             [round(offset_dx, 3), round(offset_dy, 3)],
            "anchor_overlay_b64":         anchor_overlay_b64,
            "anchors": [
                {
                    "text":                  a[4],
                    "place":                 a[5],
                    "pixel_x":               int(a[0]),
                    "pixel_y":               int(a[1]),
                    "lat":                   a[2],
                    "lng":                   a[3],
                    "reprojection_error_px": round(e, 2),
                }
                for a, e in zip(anchors_final, errors)
            ],
            "homography_params": tf.to_dict(),
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] run_georeference: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
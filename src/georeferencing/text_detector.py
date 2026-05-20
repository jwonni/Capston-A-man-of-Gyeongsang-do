"""
Hi-SAM 텍스트 영역 탐지 래퍼 (polygon JSON 출력).

Hi-SAM 저장소가 Config.HISAM_REPO_DIR에 존재해야 하며,
체크포인트가 Config.HISAM_CHECKPOINT 경로에 있어야 한다.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

try:
    import pyclipper
except ModuleNotFoundError:
    raise ImportError("pyclipper가 필요합니다: pip install pyclipper")

from src.config import Config

logger = logging.getLogger(__name__)

# 모듈 레벨에서 한 번만 로드 (요청마다 재로드 방지)
_amg = None


def _get_amg():
    global _amg
    if _amg is not None:
        return _amg

    hisam_dir = str(Config.HISAM_REPO_DIR)
    if hisam_dir not in sys.path:
        sys.path.insert(0, hisam_dir)

    from hi_sam.modeling.build import model_registry
    from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator

    args = SimpleNamespace(
        checkpoint   = str(Config.HISAM_CHECKPOINT),
        model_type   = Config.HISAM_MODEL_TYPE,
        device       = str(Config.DEVICE),
        hier_det     = True,
        input_size   = [1024, 1024],
        attn_layers  = 1,
        prompt_len   = 12,
        layout_thresh= 0.5,
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(hisam_dir)
        hisam = model_registry[Config.HISAM_MODEL_TYPE](args)
        hisam.eval().to(Config.DEVICE)
        _amg = AutoMaskGenerator(
            hisam,
            efficient_hisam=(Config.HISAM_MODEL_TYPE in ["vit_s", "vit_t"]),
        )
    finally:
        os.chdir(original_cwd)

    logger.info("Hi-SAM 모델 로드 완료 (device=%s)", Config.DEVICE)
    return _amg


def unclip(p: np.ndarray, unclip_ratio: float = 3.5) -> list:
    """polygon을 unclip_ratio 비율만큼 바깥으로 확장한다."""
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / max(poly.length, 1e-6)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(offset.Execute(distance))


def mask_to_polygon_list(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    min_area: int = 32,
) -> list[np.ndarray]:
    """binary mask → 유효 polygon 목록."""
    mask = mask.astype(np.uint8)
    polygons: list[np.ndarray] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        epsilon = 0.002 * cv2.arcLength(cont, True)
        approx  = cv2.approxPolyDP(cont, epsilon, True)
        points  = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        try:
            pts = unclip(points)
            if len(pts) != 1:
                continue
            pts = pts[0].astype(np.int32)
        except Exception:
            pts = points.astype(np.int32)
        if len(pts) < 4:
            continue
        if Polygon(pts).area < min_area:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
        polygons.append(pts)

    return polygons


def run_hisam_to_json(image_path: str, json_path: str | None = None) -> dict:
    """Hi-SAM으로 이미지에서 word polygon JSON payload를 반환한다.

    Args:
        image_path: 원본 이미지 경로
        json_path:  결과 JSON 저장 경로 (None이면 저장 생략)

    Returns:
        {image_path, image_width, image_height, num_words,
         words: [{id, vertices}, ...]}
    """
    amg = _get_amg()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    use_fp16 = Config.DEVICE.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else torch.autocast(device_type="cpu", enabled=False)

    logger.info("set_image 시작 (이미지 크기: %dx%d, fp16=%s)", img_w, img_h, use_fp16)
    with autocast_ctx:
        amg.set_image(image_rgb)
        logger.info("set_image 완료 → predict 시작")
        with torch.inference_mode():
            masks, _scores, _affinity = amg.predict(
                from_low_res    = False,
                fg_points_num   = Config.HISAM_TOTAL_POINTS,
                batch_points_num= Config.HISAM_BATCH_POINTS,
                score_thresh    = Config.HISAM_SCORE_THRESH,
                nms_thresh      = Config.HISAM_NMS_THRESH,
            )
    logger.info("predict 완료")

    if masks is None:
        raise RuntimeError("Hi-SAM 마스크 예측 실패")

    word_masks = masks[:, 0, :, :].astype(np.uint8)

    all_word_polygons: list[dict] = []
    for wm in word_masks:
        for poly in mask_to_polygon_list(wm, img_h, img_w,
                                         min_area=Config.HISAM_MIN_POLYGON_AREA):
            all_word_polygons.append({
                "id":       len(all_word_polygons),
                "vertices": poly.tolist(),
            })

    payload = {
        "image_path":   image_path,
        "image_width":  img_w,
        "image_height": img_h,
        "num_words":    len(all_word_polygons),
        "words":        all_word_polygons,
    }

    if json_path is not None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

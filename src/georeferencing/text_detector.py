"""
Hi-SAM 텍스트 영역 탐지 래퍼.

Hi-SAM 저장소가 Config.HISAM_REPO_DIR에 존재해야 하며,
체크포인트가 Config.HISAM_CHECKPOINT 경로에 있어야 한다.
"""
from __future__ import annotations

import gc
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from src.config import Config


def run_hisam(image_path: str) -> tuple[list[dict], np.ndarray]:
    """Hi-SAM으로 이미지에서 단어 영역 및 픽셀 마스크 추출.

    Returns:
        word_regions: [{cx, cy, bbox:[x1,y1,x2,y2]}, ...]  (읽기 순서 정렬)
        combined_mask: (H, W) uint8  — 전체 텍스트 영역 합산 마스크
    """
    hisam_dir = str(Config.HISAM_REPO_DIR)
    if hisam_dir not in sys.path:
        sys.path.insert(0, hisam_dir)

    from hi_sam.modeling.build import model_registry
    from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator

    args = SimpleNamespace(
        checkpoint=str(Config.HISAM_CHECKPOINT),
        model_type=Config.HISAM_MODEL_TYPE,
        device=str(Config.DEVICE),
        hier_det=True,
        input_size=[1024, 1024],
        attn_layers=1,
        prompt_len=12,
        layout_thresh=0.5,
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(hisam_dir)
        hisam = model_registry[Config.HISAM_MODEL_TYPE](args)
        hisam.eval().to(Config.DEVICE)
        amg = AutoMaskGenerator(
            hisam,
            efficient_hisam=(Config.HISAM_MODEL_TYPE in ["vit_s", "vit_t"]),
        )
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]
        amg.set_image(image_rgb)
        with torch.inference_mode():
            masks, _scores, _ = amg.predict(
                from_low_res=False,
                fg_points_num=Config.HISAM_TOTAL_POINTS,
                batch_points_num=Config.HISAM_BATCH_POINTS,
                score_thresh=Config.HISAM_SCORE_THRESH,
                nms_thresh=Config.HISAM_NMS_THRESH,
            )
    finally:
        os.chdir(original_cwd)

    if masks is None:
        raise RuntimeError("Hi-SAM 마스크 예측 실패")

    word_masks = masks[:, 0, :, :]

    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for wm in word_masks:
        combined = np.logical_or(combined, wm > 0)
    combined_mask = combined.astype(np.uint8) * 255

    # centroid 계산 시 Y축 오프셋 (텍스트 baseline → 중간 보정)
    Y_OFFSET = -13
    word_regions = []
    for wm in word_masks:
        mask_u8 = (wm > 0).astype(np.uint8)
        ys, xs  = np.where(mask_u8 > 0)
        if len(xs) == 0:
            continue
        M = cv2.moments(mask_u8)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + Y_OFFSET
        x1 = max(0,     int(xs.min()) - Config.HISAM_CROP_X_MARGIN)
        y1 = max(0,     int(ys.min()) - Config.HISAM_CROP_Y_MARGIN)
        x2 = min(img_w, int(xs.max()) + Config.HISAM_CROP_X_MARGIN)
        y2 = min(img_h, int(ys.max()) + Config.HISAM_CROP_Y_MARGIN)
        if (x2 - x1) < 10 or (y2 - y1) < 5:
            continue
        word_regions.append({"cx": cx, "cy": cy, "bbox": [x1, y1, x2, y2]})

    # 위→아래, 왼→오른 읽기 순서 정렬
    word_regions.sort(key=lambda r: (r["cy"] // 30, r["cx"]))

    del hisam, amg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return word_regions, combined_mask

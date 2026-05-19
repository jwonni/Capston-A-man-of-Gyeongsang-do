"""
PaddleOCR 기반 텍스트 인식 파이프라인.

Hi-SAM polygon JSON → 크롭 준비 → PaddleOCR → 후처리 → [{text, x, y, confidence}, ...]
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.config import Config


# ── OCR 입력 전처리 ────────────────────────────────────────────────────────────

def polygon_to_bbox(vertices: list, img_w: int, img_h: int) -> list[int]:
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    x1 = max(0,     min(xs) - Config.HISAM_CROP_X_MARGIN)
    y1 = max(0,     min(ys) - Config.HISAM_CROP_Y_MARGIN)
    x2 = min(img_w, max(xs) + Config.HISAM_CROP_X_MARGIN)
    y2 = min(img_h, max(ys) + Config.HISAM_CROP_Y_MARGIN)
    return [int(x1), int(y1), int(x2), int(y2)]


def polygon_center_xy(vertices: list, img_w: int, img_h: int) -> tuple[int, int]:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(vertices)
        if poly.is_valid and poly.area > 0:
            c  = poly.centroid
            cx = max(0, min(int(round(c.x)), img_w - 1))
            cy = max(0, min(int(round(c.y)), img_h - 1))
            return cx, cy
    except Exception:
        pass
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    cx = max(0, min(int(round(sum(xs) / len(xs))), img_w - 1))
    cy = max(0, min(int(round(sum(ys) / len(ys))), img_h - 1))
    return cx, cy


def build_crop_records_from_json(json_data: dict) -> list[dict]:
    """Hi-SAM polygon JSON → 크롭 이미지 레코드 목록."""
    image_path = json_data["image_path"]
    img        = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    records: list[dict] = []
    for item in json_data["words"]:
        vertices = item["vertices"]
        x1, y1, x2, y2      = polygon_to_bbox(vertices, img_w, img_h)
        center_x, center_y   = polygon_center_xy(vertices, img_w, img_h)
        crop = img.crop((x1, y1, x2, y2))
        records.append({
            "id":          item["id"],
            "bbox":        [x1, y1, x2, y2],
            "center_x":    center_x,
            "center_y":    center_y,
            "vertices":    vertices,
            "crop_image":  crop,
            "crop_width":  crop.width,
            "crop_height": crop.height,
        })
    return records


def adaptive_resize_keep_ratio(
    image: Image.Image,
    target_long_side: int = 256,
    min_height: int = 64,
    min_width:  int = 64,
    max_long_side: int = 1280,
) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale_candidates = [1.0]
    if max(w, h) < target_long_side:
        scale_candidates.append(target_long_side / max(w, h))
    if h < min_height:
        scale_candidates.append(min_height / h)
    if w < min_width:
        scale_candidates.append(min_width / w)
    scale = max(scale_candidates)
    if max(w, h) * scale > max_long_side:
        scale = max_long_side / max(w, h)
    if scale <= 1.0001:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def prepare_crop_for_paddle(
    crop_image: Image.Image,
    target_long_side: int = 256,
    min_height: int = 64,
    min_width:  int = 64,
    max_long_side: int = 1280,
    padding: int = 16,
) -> Image.Image:
    img = ImageOps.autocontrast(crop_image.convert("RGB"))
    img = adaptive_resize_keep_ratio(img, target_long_side, min_height, min_width, max_long_side)
    canvas = Image.new("RGB", (img.width + padding * 2, img.height + padding * 2), (255, 255, 255))
    canvas.paste(img, (padding, padding))
    return canvas


def save_prepared_crop_image(
    prepared_image: Image.Image, out_dir: str, crop_id: int, tag: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"crop_{crop_id:06d}_{tag}.png")
    prepared_image.save(out_path)
    return out_path


def build_all_paddle_inputs(crop_records: list[dict], out_dir: str) -> list[dict]:
    """크롭 레코드 → base/retry 두 해상도로 PaddleOCR 입력 이미지 준비."""
    prepared: list[dict] = []
    for rec in crop_records:
        base_img = prepare_crop_for_paddle(
            rec["crop_image"],
            target_long_side = Config.PREP_TARGET_LONG_SIDE,
            min_height       = Config.PREP_MIN_HEIGHT,
            min_width        = Config.PREP_MIN_WIDTH,
            max_long_side    = Config.PREP_MAX_LONG_SIDE,
            padding          = Config.PREP_PADDING,
        )
        retry_img = prepare_crop_for_paddle(
            rec["crop_image"],
            target_long_side = Config.PREP_RETRY_LONG_SIDE,
            min_height       = Config.PREP_MIN_HEIGHT,
            min_width        = Config.PREP_MIN_WIDTH,
            max_long_side    = Config.PREP_MAX_LONG_SIDE,
            padding          = Config.PREP_RETRY_PADDING,
        )
        base_path  = save_prepared_crop_image(base_img,  out_dir, rec["id"], "base")
        retry_path = save_prepared_crop_image(retry_img, out_dir, rec["id"], "retry")
        prepared.append({
            **rec,
            "paddle_base_path":  base_path,
            "paddle_retry_path": retry_path,
            "paddle_base_size":  list(base_img.size),
            "paddle_retry_size": list(retry_img.size),
        })
    return prepared


# ── PaddleOCR 실행 ─────────────────────────────────────────────────────────────

_paddle_ocr: Any = None


def load_paddle_ocr() -> Any:
    """PaddleOCR 싱글톤 로드 (CPU 고정)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        import paddle
        paddle.set_flags({
            "FLAGS_use_mkldnn": False,
            "FLAGS_enable_pir_api": False,
        })
        paddle.device.set_device(Config.PADDLE_DEVICE)
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(
            lang                         = Config.OCR_LANG,
            device                       = Config.PADDLE_DEVICE,
            use_doc_orientation_classify = False,
            use_doc_unwarping            = False,
            use_textline_orientation     = False,
        )
    return _paddle_ocr


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def safe_list_field(obj: Any, key: str) -> list:
    value = get_field(obj, key, None)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def unwrap_paddle_result(page_result: Any) -> Any:
    if isinstance(page_result, dict) and "res" in page_result:
        return page_result["res"]
    res_attr = getattr(page_result, "res", None)
    if res_attr is not None:
        return res_attr
    return page_result


def box_center(box_like: Any) -> tuple[float, float]:
    arr = np.asarray(box_like, dtype=np.float32)
    if arr.ndim == 1 and arr.size >= 4:
        x1, y1, x2, y2 = arr[:4]
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return 0.0, 0.0


def extract_paddle_items(page_result: Any) -> list[dict]:
    data       = unwrap_paddle_result(page_result)
    rec_texts  = safe_list_field(data, "rec_texts")
    rec_scores = safe_list_field(data, "rec_scores")
    rec_polys  = safe_list_field(data, "rec_polys")
    rec_boxes  = safe_list_field(data, "rec_boxes")
    items: list[dict] = []
    for i, text in enumerate(rec_texts):
        raw_box  = rec_boxes[i] if i < len(rec_boxes) else None
        raw_poly = rec_polys[i] if i < len(rec_polys) else None
        if isinstance(raw_box,  np.ndarray): raw_box  = raw_box.tolist()
        if isinstance(raw_poly, np.ndarray): raw_poly = raw_poly.tolist()
        geom = raw_poly if raw_poly is not None else raw_box
        cx, cy = box_center(geom) if geom is not None else (0.0, 0.0)
        score = rec_scores[i] if i < len(rec_scores) else 0.0
        if isinstance(score, np.ndarray):
            score = float(np.asarray(score).reshape(-1)[0])
        elif isinstance(score, (list, tuple)) and len(score) > 0:
            score = float(score[0])
        items.append({
            "text":  str(text).strip() if text is not None else "",
            "score": float(score) if score is not None else 0.0,
            "cx":    float(cx),
            "cy":    float(cy),
            "box":   raw_box,
            "poly":  raw_poly,
        })
    return items


def run_paddle_on_path(image_path: str, ocr: Any) -> dict:
    result = ocr.predict(input=image_path)
    if not result:
        return {"text": None, "score": 0.0, "items": []}
    page_results = result if isinstance(result, list) else [result]
    if not page_results:
        return {"text": None, "score": 0.0, "items": []}
    items     = extract_paddle_items(page_results[0])
    text      = "\n".join(it["text"] for it in items if it["text"]).strip()
    mean_score = float(np.mean([it["score"] for it in items])) if items else 0.0
    return {"text": text if text else None, "score": mean_score, "items": items}


def merge_paddle_items_to_text(items: list[dict]) -> dict:
    valid = [it for it in items if it["text"]]
    if not valid:
        return {"text": "", "score": 0.0, "num_segments": 0}
    valid = sorted(valid, key=lambda it: (it["cy"], it["cx"]))
    return {
        "text":         " ".join(it["text"] for it in valid).strip(),
        "score":        float(np.mean([it["score"] for it in valid])),
        "num_segments": len(valid),
    }


def finalize_paddle_result(raw_result: dict) -> dict:
    merged = merge_paddle_items_to_text(raw_result["items"])
    return {**raw_result, **merged}


def choose_best_paddle_result(candidates: list[dict]) -> dict:
    finalized = [finalize_paddle_result(c) for c in candidates]

    def rank_key(item: dict) -> tuple:
        text = (item.get("text") or "").strip()
        return (1 if text else 0, item.get("score", 0.0), len(text),
                -item.get("num_segments", 0))

    return max(finalized, key=rank_key)


def merge_crop_result_to_text_xy(crop_record: dict, ocr_result: dict) -> dict:
    return {
        "text":         ocr_result.get("text"),
        "x":            int(crop_record["center_x"]),
        "y":            int(crop_record["center_y"]),
        "confidence":   round(float(ocr_result.get("score", 0.0) or 0.0), 6),
        "num_segments": int(ocr_result.get("num_segments", 0) or 0),
    }


# ── OCR 텍스트 후처리 ───────────────────────────────────────────────────────────

def _clean_ocr_item(item: dict) -> dict:
    cleaned: dict = {
        "text": str(item.get("text", "")).strip(),
        "x":    int(round(float(item.get("x", 0)))),
        "y":    int(round(float(item.get("y", 0)))),
    }
    if "confidence"   in item: cleaned["confidence"]   = float(item.get("confidence",   0.0) or 0.0)
    if "num_segments" in item: cleaned["num_segments"] = int(item.get("num_segments", 0) or 0)
    return cleaned


def _merge_same_x_lines(group: list[dict]) -> dict:
    group_sorted = sorted(group, key=lambda it: (it["y"], it["x"]))
    merged: dict = {
        "text": " ".join(it["text"] for it in group_sorted),
        "x":    group_sorted[0]["x"],
        "y":    min(it["y"] for it in group_sorted),
    }
    confidences = [it["confidence"] for it in group_sorted if "confidence" in it]
    if confidences:
        merged["confidence"] = round(float(sum(confidences) / len(confidences)), 6)
    if any("num_segments" in it for it in group_sorted):
        merged["num_segments"] = int(sum(it.get("num_segments", 0) for it in group_sorted))
    return merged


def merge_lines_by_same_x(results: list[dict]) -> list[dict]:
    """X 좌표가 유사한 OCR 항목을 세로 방향으로 병합 (1회, 반복 없음)."""
    items: list[dict] = []
    for it in results:
        cleaned = _clean_ocr_item(it)
        if cleaned["text"]:
            items.append(cleaned)
    if not items:
        return []

    items_sorted = sorted(items, key=lambda it: (it["x"], it["y"]))
    groups: list[list[dict]] = []
    for item in items_sorted:
        if not groups:
            groups.append([item])
            continue
        base_x = groups[-1][0]["x"]
        last_y = groups[-1][-1]["y"]
        same_x = abs(item["x"] - base_x) <= Config.OCR_X_MERGE_TOL
        near_y = 0 <= item["y"] - last_y <= Config.OCR_Y_MERGE_TOL
        if same_x and near_y:
            groups[-1].append(item)
        else:
            groups.append([item])

    merged = [_merge_same_x_lines(g) for g in groups]
    return sorted(merged, key=lambda it: (it["y"], it["x"]))


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def run_ocr(json_data: dict, out_dir: str) -> list[dict]:
    """Hi-SAM polygon JSON → PaddleOCR → 후처리 → [{text, x, y, confidence}, ...].

    Args:
        json_data: run_hisam_to_json() 반환값
        out_dir:   전처리 크롭 이미지 저장 디렉토리

    Returns:
        confidence 필터 + 좌표 정렬 + X 병합 후처리가 완료된 텍스트 결과 목록
    """
    import gc

    import logging
    log = logging.getLogger("ocr")

    crop_records     = build_crop_records_from_json(json_data)
    prepared_records = build_all_paddle_inputs(crop_records, out_dir)
    log.info("[OCR] Hi-SAM 감지 polygon 수: %d  /  준비된 크롭 수: %d",
             json_data.get("num_words", 0), len(prepared_records))

    ocr = load_paddle_ocr()

    raw_results: list[dict] = []
    for rec in prepared_records:
        base_result  = run_paddle_on_path(rec["paddle_base_path"],  ocr)
        retry_result = run_paddle_on_path(rec["paddle_retry_path"], ocr)
        best = choose_best_paddle_result([base_result, retry_result])
        raw_results.append(merge_crop_result_to_text_xy(rec, best))

    log.info("[OCR] PaddleOCR 원시 결과 수: %d", len(raw_results))
    for r in raw_results:
        log.debug("[OCR]   raw: text=%r  conf=%.3f  x=%s  y=%s",
                  r.get("text"), float(r.get("confidence", 0) or 0),
                  r.get("x"), r.get("y"))

    # confidence 필터
    filtered = [
        r for r in raw_results
        if isinstance(r.get("text"), str)
        and r["text"].strip()
        and float(r.get("confidence", 0.0) or 0.0) >= Config.MIN_CONFIDENCE
    ]
    log.info("[OCR] confidence(>=%.2f) 필터 후: %d개", Config.MIN_CONFIDENCE, len(filtered))

    # 좌표 정렬
    if Config.SORT_BY_COORDS:
        filtered = sorted(filtered, key=lambda r: (r["y"], r["x"]))

    # X 기준 세로 병합
    merged = merge_lines_by_same_x(filtered)
    log.info("[OCR] X 병합 후 최종 결과: %d개", len(merged))
    for r in merged:
        log.info("[OCR]   최종: text=%r  conf=%.3f  x=%s  y=%s",
                 r.get("text"), float(r.get("confidence", 0) or 0),
                 r.get("x"), r.get("y"))
    return merged

'''
마라톤 코스 이미지에서 앵커 패치를 추출하고 Naver Maps Static API로 참조 타일을 다운로드하는 스크립트

- 입력: 마라톤 코스 이미지, OCR 감지 결과 JSON, 지오코딩 후보 JSON
- 출력: 앵커 패치 쌍 (소스 패치 + 참조 타일) 및 앵커 JSON

여기까지 흐름:
  1. OCR → 텍스트 후보 추출
  2. Geocoding → 텍스트 후보를 실제 좌표로 변환
  3. Map Tiles → 좌표 기준 Naver 전체 지도 이미지 다운로드
  4. 이 스크립트 → 앵커 패치 추출 + Naver Maps 참조 타일 다운로드

인증 방식 (Naver Cloud Platform Maps - Static API):
  GET https://maps.apigw.ntruss.com/map-static/v2/raster
      ?w={width}&h={height}&center={lon},{lat}&level={zoom}
  헤더: x-ncp-apigw-api-key-id / x-ncp-apigw-api-key

  NCP 콘솔 → AI·Application Service → Maps → 앱 생성 → Maps - Static 활성화
  → API Key ID / Secret을 상수에 입력
'''
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


## TODO: git에 올릴 땐, API 키는 상수에서 빈 문자열로 바꾸기
# ---------------------------------------------------------------------------
# Naver Maps Static API credentials
# ---------------------------------------------------------------------------
# 우선순위: CLI 인자 > 하드코딩 값 > 환경변수 NAVER_MAP_API_KEY_ID / NAVER_MAP_API_KEY
NAVER_MAP_API_KEY_ID_DEFAULT = "mv78ixavnb"                           # x-ncp-apigw-api-key-id
NAVER_MAP_API_KEY_DEFAULT    = "j0az183cg8DxRl1NvgBosN8whrNpBrKoTKCxSRq7"  # x-ncp-apigw-api-key

_NAVER_STATIC_URL = "https://maps.apigw.ntruss.com/map-static/v2/raster"


# ---------------------------------------------------------------------------
# 소스 패치 추출 (마라톤 이미지에서 크롭)
# ---------------------------------------------------------------------------

def crop_source_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    half = patch_size // 2

    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x1, y1 = x0 + patch_size, y0 + patch_size

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
    patch[y0c - y0:y0c - y0 + (y1c - y0c),
          x0c - x0:x0c - x0 + (x1c - x0c)] = image[y0c:y1c, x0c:x1c]

    return patch, {
        "offset_x": x0,
        "offset_y": y0,
        "center_in_image_x": float(center_x),
        "center_in_image_y": float(center_y),
        "center_in_patch_x": half,
        "center_in_patch_y": half,
        "patch_size": patch_size,
        "image_width": w,
        "image_height": h,
    }


# ---------------------------------------------------------------------------
# Naver Maps Static API 참조 타일 다운로드
# ---------------------------------------------------------------------------

def fetch_reference_tile(
    lat: float,
    lon: float,
    zoom: int,
    patch_size: int,
    api_key_id: str,
    api_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Naver Maps Static API(raster)로 patch_size × patch_size 참조 타일 다운로드.
    center 좌표가 항상 출력 이미지의 (patch_size/2, patch_size/2) 위치에 배치됨.
    """
    params = {
        "w": patch_size,
        "h": patch_size,
        "center": f"{lon},{lat}",
        "level": zoom,
    }
    headers = {
        "x-ncp-apigw-api-key-id": api_key_id,
        "x-ncp-apigw-api-key": api_key,
    }

    res = requests.get(_NAVER_STATIC_URL, params=params, headers=headers, timeout=20)
    res.raise_for_status()

    img = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Naver API 응답을 이미지로 디코딩하지 못함 (lat={lat}, lon={lon})")

    return img, {
        "zoom": zoom,
        "center_lat": lat,
        "center_lon": lon,
        "center_pixel_x": patch_size / 2.0,
        "center_pixel_y": patch_size / 2.0,
        "patch_size": patch_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="마라톤 이미지 앵커 패치 추출 + Naver Maps 참조 타일 다운로드"
    )
    parser.add_argument("--image",        required=True,
                        help="마라톤 코스 이미지 경로")
    parser.add_argument("--ocr-json",     required=True,
                        help="OCR 감지 결과 JSON (01_ocr.py 출력)")
    parser.add_argument("--geocode-json", required=True,
                        help="지오코딩 후보 JSON (02_geo_coding.py 출력)")
    parser.add_argument("--zoom",         type=int, default=16,
                        help="Naver Maps 지도 레벨 (default: 16)")
    parser.add_argument("--patch-size",   type=int, default=512,
                        help="패치 크기(px), 소스·참조 동일 (default: 512)")
    parser.add_argument("--output-dir",   default="./path-to-gpx/output/04.anchors/",
                        help="출력 디렉토리")
    parser.add_argument("--naver-api-key-id", default="",
                        help="Naver Maps API Key ID (x-ncp-apigw-api-key-id, 환경변수: NAVER_MAP_API_KEY_ID)")
    parser.add_argument("--naver-api-key",    default="",
                        help="Naver Maps API Key Secret (x-ncp-apigw-api-key, 환경변수: NAVER_MAP_API_KEY)")
    return parser


def main() -> int:
    import os

    parser = build_arg_parser()
    args = parser.parse_args()

    image_path    = Path(args.image)
    ocr_json_path = Path(args.ocr_json)
    geo_json_path = Path(args.geocode_json)

    for path, name in [
        (image_path,    "--image"),
        (ocr_json_path, "--ocr-json"),
        (geo_json_path, "--geocode-json"),
    ]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    zoom = int(args.zoom)
    if not (0 <= zoom <= 20):
        raise ValueError(f"zoom out of range (0..20): {zoom}")
    patch_size = int(args.patch_size)
    if patch_size < 32:
        raise ValueError("patch-size must be >= 32")

    # 인증 키 우선순위: CLI > 하드코딩 기본값 > 환경변수
    api_key_id = args.naver_api_key_id.strip()
    if not api_key_id:
        api_key_id = NAVER_MAP_API_KEY_ID_DEFAULT.strip() or os.environ.get("NAVER_MAP_API_KEY_ID", "").strip()

    api_key = args.naver_api_key.strip()
    if not api_key:
        api_key = NAVER_MAP_API_KEY_DEFAULT.strip() or os.environ.get("NAVER_MAP_API_KEY", "").strip()

    if not api_key_id or not api_key:
        raise ValueError(
            "Naver Maps API Key ID / Key Secret이 설정되지 않았습니다.\n"
            "  NAVER_MAP_API_KEY_ID_DEFAULT / NAVER_MAP_API_KEY_DEFAULT 상수에 입력하거나\n"
            "  --naver-api-key-id / --naver-api-key 인자로 전달하세요."
        )

    output_dir = Path(args.output_dir)
    patch_dir  = output_dir / "patches"
    for d in (output_dir, patch_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    h_img, w_img = image.shape[:2]
    print(f"[INFO] image size: {w_img}×{h_img}")

    ocr_data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    geo_data = json.loads(geo_json_path.read_text(encoding="utf-8"))

    ocr_by_idx: dict[int, dict] = {
        int(d["idx"]): d for d in ocr_data.get("detections", [])
    }

    geocoded = [
        c for c in geo_data.get("geocode_candidates", [])
        if c.get("geocoded") and c.get("lat") is not None and c.get("lon") is not None
    ]

    if not geocoded:
        raise RuntimeError(
            "geocode JSON에 geocoded 후보가 없습니다.\n"
            "02_geo_coding.py를 먼저 실행하세요."
        )

    stem = image_path.stem
    anchors: list[dict[str, Any]] = []
    print(f"[INFO] geocoded 후보 {len(geocoded)}건에 대해 앵커 패치 추출 시작")

    for cand in geocoded:
        idx  = int(cand["idx"])
        lat  = float(cand["lat"])
        lon  = float(cand["lon"])
        text = cand.get("matched_place") or cand.get("text", "")

        ocr_det = ocr_by_idx.get(idx)
        if ocr_det is None:
            print(f"  [WARN] OCR detection not found for idx={idx} ('{text}'), skipping.")
            continue

        cx = float(ocr_det["center_x"])
        cy = float(ocr_det["center_y"])
        print(f"[INFO] idx={idx} '{text}' img=({cx:.0f},{cy:.0f}) lat={lat:.6f} lon={lon:.6f}")

        src_patch, src_offset = crop_source_patch(image, cx, cy, patch_size)
        src_path = patch_dir / f"{stem}_anchor_{idx}_src.png"
        cv2.imwrite(str(src_path), src_patch)

        print(f"  [INFO] Naver Maps 참조 타일 다운로드 (zoom={zoom}) …")
        ref_tile, geotransform = fetch_reference_tile(
            lat=lat, lon=lon,
            zoom=zoom,
            patch_size=patch_size,
            api_key_id=api_key_id,
            api_key=api_key,
        )
        ref_path = patch_dir / f"{stem}_anchor_{idx}_ref.png"
        cv2.imwrite(str(ref_path), ref_tile)

        anchors.append({
            "idx":            idx,
            "text":           text,
            "lat":            lat,
            "lon":            lon,
            "display_name":   cand.get("display_name", ""),
            "image_center_x": cx,
            "image_center_y": cy,
            "src_patch":      str(src_path),
            "src_offset":     src_offset,
            "ref_tile":       str(ref_path),
            "geotransform":   geotransform,
        })
        print(f"[DONE] idx={idx} src={src_path.name} ref={ref_path.name}")

    if not anchors:
        raise RuntimeError("앵커 패치가 하나도 생성되지 않았습니다. OCR/지오코딩 결과를 확인하세요.")

    report_stem = geo_json_path.name.replace("_geocode_candidates.json", "")
    if report_stem == geo_json_path.name:
        report_stem = geo_json_path.stem

    out_json = output_dir / f"{report_stem}_anchors.json"
    out_json.write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image":          str(image_path),
        "image_width":    w_img,
        "image_height":   h_img,
        "zoom":           zoom,
        "patch_size":     patch_size,
        "anchor_count":   len(anchors),
        "anchors":        anchors,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] {len(anchors)}개 앵커 패치 → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

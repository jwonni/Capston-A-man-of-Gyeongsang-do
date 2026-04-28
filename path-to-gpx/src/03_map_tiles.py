'''
Naver Maps Static API로 지도 이미지를 다운로드하여 지오코딩 좌표의 시각적 참조를 제공하는 스크립트

- 입력: 지오코딩 후보 JSON (예: *_geocode_candidates.json)
- 출력: geocoded 후보 전체에 대한 지도 이미지 및 보고서 JSON

여기까지 흐름:
  1. OCR → 텍스트 후보 추출
  2. Geocoding → 텍스트 후보를 실제 좌표로 변환
  3. 이 스크립트 → 좌표 기준 Naver 지도 이미지 다운로드

인증 방식 (Naver Cloud Platform Maps - Static API):
  GET https://maps.apigw.ntruss.com/map-static/v2/raster-cors
      ?w={width}&h={height}&center={lon},{lat}&level={zoom}
      &X-NCP-APIGW-API-KEY-ID={API Gateway API Key ID}

  NCP 콘솔 → AI·Application Service → Maps → 앱 생성 → Maps - Static 활성화
  → API Key ID를 NAVER_MAP_API_KEY_ID_DEFAULT 상수에 입력
'''
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Naver Maps Static API credentials
# ---------------------------------------------------------------------------
# 우선순위: CLI 인자 > 하드코딩 값 > 환경변수 NAVER_MAP_API_KEY_ID / NAVER_MAP_API_KEY
NAVER_MAP_API_KEY_ID_DEFAULT = "mv78ixavnb"                           # x-ncp-apigw-api-key-id
NAVER_MAP_API_KEY_DEFAULT    = "j0az183cg8DxRl1NvgBosN8whrNpBrKoTKCxSRq7"  # x-ncp-apigw-api-key

# raster-cors 는 브라우저 img 태그용 (쿼리 파라미터 인증)
# raster 는 서버 사이드용 (헤더 인증) → Python 코드에서는 raster 사용
_NAVER_STATIC_URL = "https://maps.apigw.ntruss.com/map-static/v2/raster"

# ---------------------------------------------------------------------------
# 좌표 후보 선택
# ---------------------------------------------------------------------------

@dataclass
class CoordinateCandidate:
    idx: int
    text: str
    matched_place: str
    lat: float
    lon: float
    display_name: str



def load_all_coordinates(geocode_json_path: Path) -> list[CoordinateCandidate]:
    payload = json.loads(geocode_json_path.read_text(encoding="utf-8"))
    candidates = payload.get("geocode_candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"invalid geocode_candidates format in {geocode_json_path}")

    geocoded: list[dict[str, Any]] = [c for c in candidates if bool(c.get("geocoded", False))]
    if not geocoded:
        raise RuntimeError(
            f"no geocoded candidate found in {geocode_json_path}. Run geocoding first."
        )

    return [
        CoordinateCandidate(
            idx=int(c.get("idx", -1)),
            text=str(c.get("text", "")),
            matched_place=str(c.get("matched_place", "")),
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            display_name=str(c.get("display_name", "")),
        )
        for c in geocoded
    ]


# ---------------------------------------------------------------------------
# Naver Maps Static API 호출
# ---------------------------------------------------------------------------

def download_naver_map_image(
    lat: float,
    lon: float,
    zoom: int,
    width: int,
    height: int,
    api_key_id: str,
    api_key: str,
    tile_path: Path,
) -> dict[str, Any]:
    """
    Naver Maps Static API(raster)로 지도 이미지를 다운로드하여 tile_path에 저장.
    인증: x-ncp-apigw-api-key-id / x-ncp-apigw-api-key 헤더
    center 좌표는 항상 출력 이미지의 (width/2, height/2) 위치에 배치됨.
    """
    params = {
        "w": width,
        "h": height,
        "center": f"{lon},{lat}",
        "level": zoom,
    }
    headers = {
        "x-ncp-apigw-api-key-id": api_key_id,
        "x-ncp-apigw-api-key": api_key,
    }

    res = requests.get(_NAVER_STATIC_URL, params=params, headers=headers, timeout=20)
    res.raise_for_status()

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(res.content)

    # 저장 후 실제 이미지 크기 검증
    img = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Naver API 응답을 이미지로 디코딩하지 못함: {tile_path}")

    actual_h, actual_w = img.shape[:2]
    return {
        "zoom": zoom,
        "lat": lat,
        "lon": lon,
        "tile_path": str(tile_path),
        "saved_width": actual_w,
        "saved_height": actual_h,
        "center_pixel_in_output_x": actual_w / 2.0,
        "center_pixel_in_output_y": actual_h / 2.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch one Naver Maps Static image centered on the selected geocoding coordinate"
    )
    parser.add_argument(
        "--geocode-json",
        required=True,
        help="Path to *_geocode_candidates.json",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=17,
        help="지도 레벨 (Naver Maps level, default: 17)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1400,
        help="저장할 지도 이미지 크기(px), 가로·세로 동일 (default: 1400)",
    )
    parser.add_argument(
        "--output-dir",
        default="./path-to-gpx/output/03.map_tiles/",
        help="Directory to store map images and reports",
    )
    parser.add_argument(
        "--naver-api-key-id",
        default="",
        help="Naver Maps API Key ID (x-ncp-apigw-api-key-id, 환경변수: NAVER_MAP_API_KEY_ID)",
    )
    parser.add_argument(
        "--naver-api-key",
        default="",
        help="Naver Maps API Key Secret (x-ncp-apigw-api-key, 환경변수: NAVER_MAP_API_KEY)",
    )
    return parser


def main() -> int:
    import os

    parser = build_arg_parser()
    args = parser.parse_args()

    geocode_json_path = Path(args.geocode_json)
    if not geocode_json_path.exists() or not geocode_json_path.is_file():
        raise FileNotFoundError(f"geocode json not found: {geocode_json_path}")

    zoom = int(args.zoom)
    if not (0 <= zoom <= 20):
        raise ValueError(f"zoom out of range (0..20): {zoom}")
    tile_size = int(args.tile_size)
    if tile_size < 32:
        raise ValueError("tile-size must be >= 32")

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
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_all_coordinates(geocode_json_path)
    print(f"[INFO] geocoded 후보 {len(candidates)}건에 대해 타일 다운로드 시작")

    report_stem = geocode_json_path.name.replace("_geocode_candidates.json", "")
    if report_stem == geocode_json_path.name:
        report_stem = geocode_json_path.stem

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_results: list[dict[str, Any]] = []
    for candidate in candidates:
        place_label = candidate.matched_place or candidate.text
        tile_path = tiles_dir / f"{report_stem}_idx{candidate.idx}_z{zoom}_naver.png"
        print(
            f"[INFO] idx={candidate.idx} '{place_label}' "
            f"lat={candidate.lat:.7f}, lon={candidate.lon:.7f} → {tile_path.name}"
        )
        tile_result = download_naver_map_image(
            lat=candidate.lat,
            lon=candidate.lon,
            zoom=zoom,
            width=tile_size,
            height=tile_size,
            api_key_id=api_key_id,
            api_key=api_key,
            tile_path=tile_path,
        )
        tile_result["idx"] = candidate.idx
        tile_result["place_label"] = place_label
        tile_results.append(tile_result)
        print(
            f"[DONE] idx={candidate.idx} saved={tile_path.name} "
            f"size={tile_result['saved_width']}x{tile_result['saved_height']}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_geocode_json": str(geocode_json_path),
        "zoom": zoom,
        "tile_size": tile_size,
        "tile_count": len(tile_results),
        "tiles": tile_results,
        "candidates": [asdict(c) for c in candidates],
    }
    report_path = output_dir / f"{report_stem}_map_tiles_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

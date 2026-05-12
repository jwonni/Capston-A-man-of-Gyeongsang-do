"""
카카오 Local API를 이용한 앵커 구축.

앵커 튜플 형식: (pixel_x, pixel_y, lat, lng, text, place)
"""
from __future__ import annotations

import requests

from src.config import Config


def kakao_search(query: str, api_key: str) -> tuple[float, float, str] | None:
    """카카오 키워드 검색 API → (lat, lng, place_name)."""
    url     = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params  = {"query": query, "size": 1}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        docs = resp.json().get("documents", [])
        if not docs:
            return None
        return float(docs[0]["y"]), float(docs[0]["x"]), docs[0]["place_name"]
    except Exception:
        return None


def is_good_anchor(text: str) -> bool:
    """앵커로 사용 가능한 텍스트인지 판단."""
    if len(text) < Config.ANCHOR_MIN_TEXT_LEN:
        return False
    if text in Config.ANCHOR_BLACKLIST:
        return False
    for ww in Config.ANCHOR_WHITELIST:
        if ww in text:
            return True
    return True


def build_raw_anchors(
    ocr_results: list[dict],
    api_key: str,
) -> list[tuple]:
    """OCR 결과 → 카카오 API 조회 → (pixel_x, pixel_y, lat, lng, text, place) 목록.

    블랙리스트/화이트리스트 필터 후 카카오 API로 유효한 결과만 반환한다.
    """
    candidates = [r for r in ocr_results if is_good_anchor(r["text"])]
    raw = []
    for item in candidates:
        result = kakao_search(item["text"], api_key)
        if result:
            lat, lng, place = result
            raw.append((item["x"], item["y"], lat, lng, item["text"], place))
    return raw

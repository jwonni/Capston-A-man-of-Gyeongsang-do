'''
지오코딩을 수행하여 OCR 후보 텍스트를 실제 지리적 위치로 변환하는 스크립트

조회 우선순위:
  1. marathon_landmarks.json (정밀 DB)
  2. Kakao Local Search API (한국 지명 키워드 검색)
  3. Nominatim (OpenStreetMap) 폴백

- 입력: OCR 감지 결과 JSON (예: *_ocr_detections.json)
- 출력: 지오코딩 후보 JSON (예: *_geocode_candidates.json)

인증 방식 (Kakao Developers - 로컬 키워드 검색 API):
  GET https://dapi.kakao.com/v2/local/search/keyword.json?query={장소명}
  Authorization: KakaoAK {REST_API_KEY}

  Kakao Developers → 내 애플리케이션 → 앱 생성 → REST API 키 복사
  → 아래 KAKAO_REST_API_KEY_DEFAULT 상수에 입력
'''
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from geopy.geocoders import Nominatim


# ---------------------------------------------------------------------------
# Kakao API credentials
# ---------------------------------------------------------------------------
# 우선순위: CLI 인자 > 하드코딩 값 > 환경변수 KAKAO_REST_API_KEY
KAKAO_REST_API_KEY_DEFAULT = "dc28dc8ef2e29162ab6643fcd2aa43a5"

_KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


# ---------------------------------------------------------------------------
# Kakao Local Search API 호출
# ---------------------------------------------------------------------------

def _kakao_geocode(query_text: str, api_key: str) -> dict[str, Any] | None:
    """
    Kakao 로컬 키워드 검색 API 호출.
    헤더: Authorization: KakaoAK {REST_API_KEY}
    HTTP 오류 발생 시 requests.HTTPError를 그대로 전파 (caller에서 처리).
    """
    res = requests.get(
        _KAKAO_KEYWORD_URL,
        headers={"Authorization": f"KakaoAK {api_key}"},
        params={"query": query_text},
        timeout=15,
    )
    res.raise_for_status()

    documents = res.json().get("documents", [])
    if not documents:
        return None

    doc = documents[0]
    return {
        "lat": float(doc["y"]),
        "lon": float(doc["x"]),
        "display_name": doc.get("place_name") or doc.get("address_name") or query_text,
    }


# ---------------------------------------------------------------------------
# HTTP 오류 진단
# ---------------------------------------------------------------------------

def _diagnose_http_error(exc: requests.exceptions.HTTPError) -> str:
    """HTTP 오류 응답 body를 읽어 원인별 진단 메시지 반환."""
    try:
        body = exc.response.text.strip()
    except Exception:
        body = "(응답 body 읽기 실패)"

    code = exc.response.status_code if exc.response is not None else "?"

    if code == 401:
        return (
            f"  응답: {body}\n"
            f"  확인: Kakao Developers에서 REST API 키를 재확인하세요."
        )
    if code == 403:
        return (
            f"  응답: {body}\n"
            f"  확인: Kakao Developers → 앱 설정 → 플랫폼에서 허용 도메인/IP를 등록하세요."
        )
    return f"  응답: {body}"


# ---------------------------------------------------------------------------
# 랜드마크 DB 로드
# ---------------------------------------------------------------------------

def load_landmark_db() -> list[dict]:
    """marathon_landmarks.json 로드. 없으면 빈 리스트 반환."""
    db_path = Path(__file__).resolve().parent / "marathon_landmarks.json"
    if not db_path.exists():
        print(f"[WARN] landmark DB not found: {db_path}  → API-only mode")
        return []
    data = json.loads(db_path.read_text(encoding="utf-8"))
    return data.get("entries", [])


def lookup_landmark_db(query_text: str, db: list[dict]) -> dict | None:
    """query_text가 DB alias와 완전히 동일할 때만 entry 반환."""
    for entry in db:
        for alias in entry.get("aliases", []):
            if alias == query_text:
                return entry
    return None


# ---------------------------------------------------------------------------
# GPS 좌표 정규식 추출
# ---------------------------------------------------------------------------

def extract_geo_candidates(text: str) -> dict[str, Any]:
    gps_matches: list[dict[str, Any]] = []
    decimal_pattern = re.compile(r"(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")
    for m in decimal_pattern.finditer(text):
        lat = float(m.group(1))
        lon = float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            gps_matches.append({
                "raw": m.group(0), "lat": lat, "lon": lon, "source": "regex_decimal"
            })
    return {"gps_matches": gps_matches}


# ---------------------------------------------------------------------------
# 지오코딩 파이프라인 (DB → Kakao → Nominatim)
# ---------------------------------------------------------------------------

def geocode_text_candidates(
    detections: list[dict[str, Any]],
    top_k: int,
    db: list[dict],
    kakao_api_key: str,
) -> list[dict[str, Any]]:
    geolocator = Nominatim(user_agent="marathon_gpx_pipeline/1.0")

    sorted_dets = sorted(
        detections,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )[:top_k]

    out: list[dict[str, Any]] = []

    for det in sorted_dets:
        text_raw      = str(det.get("text", "")).strip()
        matched_place = str(det.get("matched_place", "")).strip()
        query_text    = matched_place if matched_place else text_raw

        if len(query_text) < 2:
            continue

        base = {
            "idx": int(det.get("idx", -1)),
            "text": text_raw,
            "matched_place": matched_place,
        }

        # ── 1단계: 랜드마크 DB ────────────────────────────────────────────
        entry = lookup_landmark_db(query_text, db)
        if entry is not None:
            out.append({
                **base,
                "geocoded": True,
                "lat": entry["lat"],
                "lon": entry["lon"],
                "display_name": entry.get("note", entry["canonical"]),
                "source": "landmark_db",
                "db_canonical": entry["canonical"],
            })
            print(
                f"  [DB]    idx={det.get('idx')}  '{matched_place or text_raw}'"
                f" → {entry['canonical']}  ({entry['lat']:.5f}, {entry['lon']:.5f})"
            )
            continue

        # ── 2단계: Kakao Geocoding ───────────────────────────────────────
        if kakao_api_key:
            kakao = None
            try:
                kakao = _kakao_geocode(query_text=query_text, api_key=kakao_api_key)
                time.sleep(0.15)
            except requests.exceptions.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "?"
                reason = exc.response.reason if exc.response is not None else ""
                print(f"  [KAKAO] idx={det.get('idx')}  '{query_text}' → HTTP {code} {reason}")
                diag = _diagnose_http_error(exc)
                for line in diag.splitlines():
                    if line.strip():
                        print(f"          {line}")
            except Exception as exc:
                print(f"  [KAKAO] idx={det.get('idx')}  '{query_text}' → 오류: {exc}")

            if kakao is not None:
                out.append({
                    **base,
                    "geocoded": True,
                    "lat": kakao["lat"],
                    "lon": kakao["lon"],
                    "display_name": kakao["display_name"],
                    "source": "kakao",
                })
                print(
                    f"  [KAKAO] idx={det.get('idx')}  '{query_text}'"
                    f" → {kakao['lat']:.5f}, {kakao['lon']:.5f}"
                    f"  ({kakao['display_name']})"
                )
                continue
            else:
                print(
                    f"  [KAKAO] idx={det.get('idx')}  '{query_text}'"
                    f" → 결과 없음 (Nominatim 폴백)"
                )

        # ── 3단계: Nominatim 폴백 ─────────────────────────────────────────
        try:
            location = geolocator.geocode(
                query_text,
                addressdetails=True,
                country_codes="kr",
                language="ko",
            )
            time.sleep(1.0)
        except Exception as exc:
            out.append({**base, "geocoded": False, "source": "nominatim", "reason": str(exc)})
            continue

        if location is None:
            out.append({**base, "geocoded": False, "source": "nominatim", "reason": "not found"})
            print(f"  [NOM]   idx={det.get('idx')}  '{query_text}' → not found")
            continue

        out.append({
            **base,
            "geocoded": True,
            "lat": float(location.latitude),
            "lon": float(location.longitude),
            "display_name": location.address,
            "source": "nominatim",
        })
        print(
            f"  [NOM]   idx={det.get('idx')}  '{query_text}'"
            f" → {location.latitude:.5f}, {location.longitude:.5f}"
        )

    return out


# ---------------------------------------------------------------------------
# 파이프라인 진입점
# ---------------------------------------------------------------------------

def run_geo_coding(
    ocr_json_path: Path,
    output_dir: Path | None,
    top_k: int,
    kakao_api_key: str,
) -> Path:
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    detections = data.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError(f"invalid detections format in: {ocr_json_path}")

    db = load_landmark_db()
    db_count = len(db)
    print(f"[INFO] landmark DB: {db_count} entries loaded")

    if kakao_api_key:
        print(f"[INFO] Kakao REST API 키 확인됨 ({kakao_api_key[:6]}...)")
        print("[INFO] geocoder 우선순위: landmark_db → kakao → nominatim")
    else:
        print("[WARN] Kakao API 키 없음 → DB hit 외에는 Nominatim 사용")

    full_text = " ".join(str(d.get("text", "")) for d in detections)
    gps_candidates = extract_geo_candidates(full_text)

    print(f"[INFO] geocoding {min(len(detections), top_k)}건 처리 중 (top_k={top_k}) …")
    geocode_results = geocode_text_candidates(
        detections,
        top_k=max(1, top_k),
        db=db,
        kakao_api_key=kakao_api_key,
    )

    db_hits    = sum(1 for r in geocode_results if r.get("source") == "landmark_db")
    kakao_hits = sum(1 for r in geocode_results if r.get("source") == "kakao" and r.get("geocoded"))
    nom_hits   = sum(1 for r in geocode_results if r.get("source") == "nominatim" and r.get("geocoded"))
    print(f"[INFO] 결과: landmark_db={db_hits}, kakao={kakao_hits}, nominatim={nom_hits}")

    image_path = str(data.get("image", ""))
    out_dir = output_dir if output_dir is not None else ocr_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = ocr_json_path.name
    stem = stem[: -len("_ocr_detections.json")] if stem.endswith("_ocr_detections.json") else ocr_json_path.stem

    out_path = out_dir / f"{stem}_geocode_candidates.json"
    out_path.write_text(json.dumps({
        "image": image_path,
        "mode": data.get("mode", "kr_place_whitelist_only"),
        "source_detection_json": str(ocr_json_path),
        "landmark_db_entries": db_count,
        "geocode_candidates": geocode_results,
        "gps_regex_candidates": gps_candidates,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Geocode OCR detections: landmark DB → Kakao → Nominatim."
    )
    parser.add_argument("--ocr-json", required=True,
                        help="입력 *_ocr_detections.json 경로")
    parser.add_argument("--output-dir", default="./path-to-gpx/output/02.geocoding/",
                        help="출력 디렉토리")
    parser.add_argument("--top-k", type=int, default=10,
                        help="지오코딩할 최대 감지 수 (기본: 10)")
    parser.add_argument("--kakao-api-key", default="",
                        help="Kakao REST API 키 (환경변수: KAKAO_REST_API_KEY)")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json)
    if not ocr_json_path.exists() or not ocr_json_path.is_file():
        raise FileNotFoundError(f"ocr detection json not found: {ocr_json_path}")

    # 인증 키 우선순위: CLI > 하드코딩 기본값 > 환경변수
    kakao_api_key = args.kakao_api_key.strip()
    if not kakao_api_key:
        kakao_api_key = KAKAO_REST_API_KEY_DEFAULT.strip() or os.environ.get("KAKAO_REST_API_KEY", "").strip()

    out_dir = Path(args.output_dir) if args.output_dir else None
    out_path = run_geo_coding(
        ocr_json_path=ocr_json_path,
        output_dir=out_dir,
        top_k=args.top_k,
        kakao_api_key=kakao_api_key,
    )
    print(f"[DONE] geocode json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

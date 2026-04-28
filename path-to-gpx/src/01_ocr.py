"""OCR + place filtering for geocoding candidates.

Pipeline:
1) Read text boxes from PaddleOCR (fallback: EasyOCR)
2) Keep only entries that match Korean place whitelist and confidence threshold
3) Export detections with geometry (center/size/normalized center)

Primary outputs:
- *_ocr_visualized.png
- *_ocr_detections.json
- *_ocr_detections.csv
- *_ocr_rejected.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[PaddleOCR 실패 원인 분석 — v2.x → v3.x 브레이킹 체인지]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 생성자 파라미터 변경
   - 구 코드: PaddleOCR(use_textline_orientation=True, lang='korean')
   - 신 API: use_textline_orientation 은 __init__ 에 없고 predict() 에만 존재.
     lang 파라미터는 유지되나 내부 처리 방식이 변경됨.
   - 구 코드가 use_textline_orientation=True 를 __init__ 에 넘기면
     **kwargs 로 흡수되어 무시되거나 TypeError 발생.

② .ocr() 메서드 deprecated
   - v3.x 에서 ocr() 는 @deprecated 데코레이터로 감싸져 있고
     내부적으로 predict() 를 호출하는 shim 으로만 남아 있음.
   - cls=True 같은 구 파라미터는 **kwargs 로 전달되나 실제로 사용되지 않음.

③ 반환값 구조 완전 교체
   - 구 API: list[list[tuple]] — [[box, (text, score)], ...]
   - 신 API: list[OCRResult(dict-like)] — 각 item 이 dict 서브클래스이며
     접근 키: "rec_polys", "rec_texts", "rec_scores"
     (구 코드가 체크하던 "dt_polys" 는 내부 중간 결과용 키로 남아 있음)
   - 구 코드의 isinstance(result[0], dict) 분기는 올바르게 진입하지만
     "dt_polys" 키 대신 "rec_polys" 를 써야 함.

④ 네트워크 의존성 — 모델 다운로드 필수
   - v3.x 는 초기화 시 HuggingFace / ModelScope / AIStudio 중 하나에서
     모델 가중치를 자동 다운로드함.
   - 오프라인/방화벽 환경에서는 Exception 이 발생해 EasyOCR 폴백으로 넘어감.
   - 환경변수 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True 로 체크를 건너뛰고
     로컬 캐시만 사용하도록 강제할 수 있음.

⑥ oneDNN(MKL-DNN) 미지원 AttributeType — 실제 발생 오류
   - 오류: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
           [pir::ArrayAttribute<pir::DoubleAttribute>]
           at onednn_instruction.cc:118
   - 원인: PaddlePaddle PIR 새 실행기의 oneDNN 백엔드가 DoubleAttribute 배열
     타입 변환을 구현하지 않아 CPU 추론 시 항상 예외 발생.
   - 해결: FLAGS_use_mkldnn=0 환경변수를 paddle import 전에 설정하여
     oneDNN 커널 대신 일반 CPU 커널을 사용하도록 강제.

⑤ predict() 호출 시그니처
   - 구 코드: ocr.predict(image_path) — 위치 인자로 전달
   - 신 API: predict(input, *, use_textline_orientation=None, ...)
     키워드 전용 옵션이므로 image_path 는 그대로 써도 되지만
     use_textline_orientation 는 반드시 predict() 에 넘겨야 함.

수정 전략:
  - _try_build_paddle_ocr : 생성자에서 use_textline_orientation 제거
  - _run_paddle_ocr       : predict() 사용 + 반환 키를 rec_polys/rec_texts/rec_scores 로 통일
  - 구 list-of-list 파싱 분기(item[0]/item[1][0]/item[1][1])는 삭제
  - 환경변수 주입으로 오프라인 안정성 확보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ─── 환경변수: PaddleOCR 3.x 의 네트워크 연결 체크 비활성화 ───────────────
# 이 값이 없으면 초기화마다 HuggingFace 등에 접속을 시도하며,
# 오프라인 환경에서는 Exception 이 발생해 폴백으로 빠짐.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ─── 환경변수: oneDNN(MKL-DNN) 비활성화 ──────────────────────────────────
# PaddlePaddle PIR 실행기의 oneDNN 백엔드가 ArrayAttribute<DoubleAttribute>
# 타입 변환을 지원하지 않아 CPU 추론 시 오류 발생.
# paddle import 전에 반드시 설정해야 적용됨.
os.environ["FLAGS_use_mkldnn"] = "0"


# ─── 데이터 클래스 ─────────────────────────────────────────────────────────

@dataclass
class OCRDetection:
    idx: int
    text: str
    confidence: float
    matched_place: str
    polygon_xy: list[list[float]]
    center_x: float
    center_y: float
    width: float
    height: float
    norm_center_x: float
    norm_center_y: float


# ─── 한국 장소 화이트리스트 ────────────────────────────────────────────────

KR_PLACE_KEYWORDS: set[str] = {
    "여의도한강공원",
    "남산",
    "남산타워",
    "롯데타워",
    "광화문",
    "경복궁",
    "청와대",
    "해운대",
    "광안리",
    "제주공항",
    "김포공항",
    "인천공항",
    "잠실역",
    "강남역",
    "홍대입구역",
    "신촌역",
    "국회의사당역",
    "국회의사당",
    "서울숲",
    "여의도역",
    "광화문역",
    "종로3가역",
    "명동역",
    "동대문역사문화공원역",
    "여의도공원 문화의마당",
    "더현대서울",
    "서울역",
    "부산역",
    "대구역",
    "인천역",
    "광주역",
    "대전역",
    "울산역",
    "세종역",
    "수원역",
    "성남역",
    "용인역",
    "고양역",
    "부천역",
    "안양역",
    "창원역",
    "포항역",
    "경주역",
    "전주역",
    "천안역",
    "청주역",
    "한강역",
    "나주시청",
    "나주대교",
    "빛가람대교",
    "카이스트",
    "엑스포다리",
    "한밭수목원",
    "마포구청역",
    "난지천공원",
    "광화문광장",
    "마포대교",
    "양화대교",
    "당산역",
    "63빌딩",
    "충정로역",
    "덕수궁",
    "안동시민운동장",
    "대구지방검찰청",
    "대구지방법원",
    "낙천공원",
    "영호대교",
    "만년고등학교",
    "정부대전청사"
}

# 정규화된 키워드 캐시 (긴 키워드 우선으로 정렬하여 부분 매칭 신뢰도 향상)
_NORMALIZED_KEYWORDS: list[tuple[str, str]] = [
    (keyword, re.sub(r"[^0-9a-zA-Z가-힣]", "", keyword).lower())
    for keyword in sorted(KR_PLACE_KEYWORDS, key=len, reverse=True)
]


# ─── 텍스트 처리 유틸 ──────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """한/영/숫자만 남기고 소문자 변환."""
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", s).lower()


def match_kr_place(text: str) -> str:
    """화이트리스트 키워드 중 텍스트에 포함된 것 반환. 없으면 빈 문자열."""
    norm_text = normalize_text(text)
    if not norm_text:
        return ""
    for keyword, norm_key in _NORMALIZED_KEYWORDS:
        if norm_key and norm_key in norm_text:
            return keyword
    return ""


# ─── OCR Detection 생성 헬퍼 ──────────────────────────────────────────────

def _new_detection(
    idx: int,
    text: str,
    confidence: float,
    polygon_xy: list[list[float]],
) -> OCRDetection:
    return OCRDetection(
        idx=idx,
        text=str(text),
        confidence=float(confidence),
        matched_place="",
        polygon_xy=polygon_xy,
        center_x=0.0,
        center_y=0.0,
        width=0.0,
        height=0.0,
        norm_center_x=0.0,
        norm_center_y=0.0,
    )


# ─── PaddleOCR 3.x 래퍼 ───────────────────────────────────────────────────

def _try_build_paddle_ocr(lang: str) -> Any:
    """
    PaddleOCR 3.x 생성자.

    [수정 포인트 ①]
    구 코드: PaddleOCR(use_textline_orientation=True, lang=lang)
    → use_textline_orientation 은 v3.x __init__ 파라미터가 아님.
      predict() 호출 시에 전달해야 하므로 생성자에서 제거.
    """
    from paddleocr import PaddleOCR
    return PaddleOCR(lang=lang)


def _run_paddle_ocr(ocr: Any, image_path: str) -> list[OCRDetection]:
    """
    PaddleOCR 3.x predict() 호출 및 결과 파싱.

    [수정 포인트 ②] predict() 호출 방식
    - 구 코드: ocr.ocr(image_path, cls=True)  → deprecated shim
    - 신 코드: ocr.predict(image_path, use_textline_orientation=True)
      use_textline_orientation 은 predict() 의 키워드 전용 인자임.

    [수정 포인트 ③] 반환값 구조 (OCRResult — dict 서브클래스)
    - 구 코드가 접근하던 키 "dt_polys" 는 내부 중간 결과용.
      공개 접근 키: "rec_polys", "rec_texts", "rec_scores"
    - 구 list-of-list 분기(item[0]/item[1][0]/item[1][1])는 v3.x 에서 불필요.
    """
    result: list[Any] = ocr.predict(
        image_path,
        use_textline_orientation=True,  # 수정 포인트 ①·②
    )

    detections: list[OCRDetection] = []
    idx = 1

    if not result:
        return detections

    for item in result:
        # item 은 OCRResult (dict-like).
        # 수정 포인트 ③: rec_polys / rec_texts / rec_scores 사용
        polys  = item.get("rec_polys")   # shape: (N, 4, 2)
        texts  = item.get("rec_texts")   # list[str]
        scores = item.get("rec_scores")  # list[float]

        if polys is None or texts is None or scores is None:
            print(f"[WARN] PaddleOCR 결과에 예상 키가 없음: {list(item.keys())}")
            continue

        for poly, text, score in zip(polys, texts, scores):
            polygon = [[float(p[0]), float(p[1])] for p in np.array(poly)]
            detections.append(
                _new_detection(idx=idx, text=str(text), confidence=float(score), polygon_xy=polygon)
            )
            idx += 1

    return detections


# ─── EasyOCR 폴백 ─────────────────────────────────────────────────────────

def _run_easyocr(image_path: str, langs: list[str]) -> list[OCRDetection]:
    import easyocr
    reader = easyocr.Reader(langs, gpu=False)
    result = reader.readtext(image_path, detail=1, paragraph=False)

    detections: list[OCRDetection] = []
    for idx, item in enumerate(result, start=1):
        box, text, conf = item[0], item[1], float(item[2])
        polygon = [[float(p[0]), float(p[1])] for p in box]
        detections.append(_new_detection(idx=idx, text=text, confidence=conf, polygon_xy=polygon))

    return detections


# ─── 엔진 선택 진입점 ─────────────────────────────────────────────────────

def run_ocr(image_path: Path, lang: str = "korean") -> tuple[str, list[OCRDetection]]:
    """PaddleOCR 3.x 우선 시도, 실패 시 EasyOCR 폴백."""
    paddle_lang = "korean" if lang in ("korean", "default") else "en"
    easy_langs  = ["ko", "en"] if paddle_lang == "korean" else ["en"]

    try:
        ocr = _try_build_paddle_ocr(paddle_lang)
        detections = _run_paddle_ocr(ocr, str(image_path))
        print(f"[INFO] OCR 엔진: PaddleOCR (lang={paddle_lang})")
        return "paddleocr", detections
    except Exception as exc:
        print(f"[WARN] PaddleOCR 실패 — EasyOCR 폴백: {exc}")

    try:
        detections = _run_easyocr(str(image_path), easy_langs)
        print(f"[INFO] OCR 엔진: EasyOCR (langs={easy_langs})")
        return "easyocr", detections
    except Exception as exc:
        raise RuntimeError(
            "사용 가능한 OCR 엔진이 없습니다. "
            "'paddleocr' (권장) 또는 'easyocr' 를 설치하세요."
        ) from exc


# ─── 후처리: 지오메트리 보강 ──────────────────────────────────────────────

def enrich_geometry(
    detections: list[OCRDetection], width: int, height: int
) -> list[OCRDetection]:
    """바운딩 박스 기반 geometry 필드(center/size/norm) 채우기."""
    for det in detections:
        pts = np.array(det.polygon_xy, dtype=np.float32)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)

        det.width    = float(max_xy[0] - min_xy[0])
        det.height   = float(max_xy[1] - min_xy[1])
        det.center_x = float((min_xy[0] + max_xy[0]) / 2.0)
        det.center_y = float((min_xy[1] + max_xy[1]) / 2.0)
        det.norm_center_x = det.center_x / max(width, 1)
        det.norm_center_y = det.center_y / max(height, 1)
    return detections


# ─── 후처리: 화이트리스트 필터링 ─────────────────────────────────────────

def filter_detections_to_kr_places(
    detections: list[OCRDetection],
    min_confidence: float,
) -> tuple[list[OCRDetection], list[dict[str, Any]]]:
    filtered: list[OCRDetection] = []
    rejected: list[dict[str, Any]] = []

    for det in detections:
        matched = match_kr_place(det.text)
        det.matched_place = matched
        if det.confidence >= min_confidence and matched:
            filtered.append(det)
        else:
            rejected.append({
                "idx": det.idx,
                "text": det.text,
                "confidence": det.confidence,
                "reason": "below_confidence_or_not_whitelisted",
            })

    return filtered, rejected


# ─── 후처리: 장소별 중복 제거 ─────────────────────────────────────────────

def deduplicate_by_place(
    detections: list[OCRDetection],
) -> tuple[list[OCRDetection], list[dict[str, Any]]]:
    """동일 장소(matched_place) 중 confidence × bbox_area 가장 높은 대표 1개만 유지."""
    groups: dict[str, list[OCRDetection]] = defaultdict(list)
    for det in detections:
        groups[det.matched_place].append(det)

    kept: list[OCRDetection] = []
    dedup_rejected: list[dict[str, Any]] = []

    for _, group in groups.items():
        best = max(group, key=lambda d: d.confidence * max(d.width * d.height, 1.0))
        kept.append(best)
        for det in group:
            if det is not best:
                dedup_rejected.append({
                    "idx": det.idx,
                    "text": det.text,
                    "matched_place": det.matched_place,
                    "confidence": det.confidence,
                    "reason": f"duplicate_place_dedup (kept idx={best.idx})",
                })

    if dedup_rejected:
        print(
            f"[INFO] dedup: {len(dedup_rejected)}개 중복 제거 "
            f"({len(kept)}개 고유 장소 유지)"
        )
    return kept, dedup_rejected


# ─── 시각화 ───────────────────────────────────────────────────────────────

def draw_visualization(
    image: np.ndarray, detections: list[OCRDetection]
) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        pts = np.array(det.polygon_xy, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"{det.idx}:{det.text[:24]} ({det.confidence:.2f})"
        tx = int(max(0, det.center_x - 40))
        ty = int(max(20, det.center_y - 8))
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


# ─── 출력 저장 ─────────────────────────────────────────────────────────────

def save_outputs(
    image_path: Path,
    output_dir: Path,
    engine_name: str,
    detections: list[OCRDetection],
    rejected_detections: list[dict[str, Any]],
    vis_image: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    vis_path      = output_dir / f"{stem}_ocr_visualized.png"
    json_path     = output_dir / f"{stem}_ocr_detections.json"
    csv_path      = output_dir / f"{stem}_ocr_detections.csv"
    rejected_path = output_dir / f"{stem}_ocr_rejected.json"

    cv2.imwrite(str(vis_path), vis_image)

    json_data = {
        "image": str(image_path),
        "engine": engine_name,
        "mode": "kr_place_whitelist_only",
        "whitelist_size": len(KR_PLACE_KEYWORDS),
        "detection_count": len(detections),
        "rejected_count": len(rejected_detections),
        "detections": [
            {
                "idx": d.idx,
                "text": d.text,
                "matched_place": d.matched_place,
                "confidence": d.confidence,
                "polygon_xy": d.polygon_xy,
                "center_x": d.center_x,
                "center_y": d.center_y,
                "width": d.width,
                "height": d.height,
                "norm_center_x": d.norm_center_x,
                "norm_center_y": d.norm_center_y,
            }
            for d in detections
        ],
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "text", "matched_place", "confidence",
            "center_x", "center_y", "width", "height",
            "norm_center_x", "norm_center_y", "polygon_xy",
        ])
        for d in detections:
            writer.writerow([
                d.idx, d.text, d.matched_place, d.confidence,
                d.center_x, d.center_y, d.width, d.height,
                d.norm_center_x, d.norm_center_y,
                json.dumps(d.polygon_xy, ensure_ascii=False),
            ])

    rejected_path.write_text(
        json.dumps(rejected_detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[DONE] visualization : {vis_path}")
    print(f"[DONE] detection json: {json_path}")
    print(f"[DONE] detection csv : {csv_path}")
    print(f"[DONE] rejected json : {rejected_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="이미지에서 텍스트를 추출하고 한국 지명을 필터링하여 지오코딩용 좌표를 저장합니다."
    )
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument(
        "--output-dir",
        default="./path-to-gpx/output/01.ocr/",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--lang",
        default="korean",
        choices=["korean", "english"],
        help="OCR 언어 프리셋",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="KR 화이트리스트 필터 후 유지할 최소 OCR 신뢰도",
    )
    parser.add_argument(
        "--geocode-top-k",
        type=int,
        default=None,
        help="[deprecated] 지오코딩은 geo_coding.py --top-k 를 사용하세요.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.geocode_top_k is not None:
        print("[WARN] '--geocode-top-k' 는 deprecated. geo_coding.py --top-k 를 사용하세요.")

    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없음: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    h, w = image.shape[:2]

    # 1) OCR
    engine_name, detections = run_ocr(image_path=image_path, lang=args.lang)

    # 2) Geometry 보강 + 화이트리스트/신뢰도 필터
    detections = enrich_geometry(detections, width=w, height=h)
    detections, rejected = filter_detections_to_kr_places(
        detections=detections,
        min_confidence=max(0.0, min(1.0, args.min_confidence)),
    )
    print(f"[INFO] kept={len(detections)} (KR whitelist), rejected={len(rejected)}")

    # 3) 장소별 중복 제거 (TPS 안정화)
    detections, dedup_rejected = deduplicate_by_place(detections)
    rejected.extend(dedup_rejected)

    if not detections:
        print("[WARN] 화이트리스트 장소가 감지되지 않았습니다. KR_PLACE_KEYWORDS 를 확인하세요.")

    vis = draw_visualization(image, detections)
    save_outputs(
        image_path=image_path,
        output_dir=Path(args.output_dir),
        engine_name=engine_name,
        detections=detections,
        rejected_detections=rejected,
        vis_image=vis,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
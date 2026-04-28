"""06_tps_align.py — LoFTR 특징점 기반 박판 스플라인(TPS) 모델

LoFTR 매칭된 특징점들의 픽셀 좌표와 실제 지리 좌표(lat, lon)의 대응관계로
박판 스플라인(TPS) 모델을 구축합니다.

이 모델은 마라톤 도면 이미지의 모든 픽셀(px, py)을 실제 좌표(lat, lon)로
정확하게 변환하는 데 사용됩니다.

입력
----
--loftr-json     06_loftr_match.py의 출력 (LoFTR 특징점)
--smoothing      RBF 평활화 매개변수 (기본값: 1e-4)
--min-dist-px    중복 제거용 최소 픽셀 거리 (기본값: 10.0)
--output-dir     출력 디렉토리 (기본값: ./path-to-gpx/output/06.tps/)

출력
----
<output-dir>/<stem>_tps_model.json — LoFTR 특징점 + 품질 지표
                                    (TPS는 런타임에 특징점에서 다시 적합됨)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 제어점 수집 - LoFTR만 사용
# ---------------------------------------------------------------------------

def _loftr_cps(loftr_data: dict) -> list[dict]:
    """06_loftr_match.py 출력에서 LoFTR 기반 제어점을 추출합니다."""
    cps = []
    for res in loftr_data.get("anchor_results", []):
        if res.get("status") != "ok":
            continue
        for cp in res.get("control_points", []):
            cps.append({
                "img_x":  cp["img_x"],
                "img_y":  cp["img_y"],
                "lat":    cp["lat"],
                "lon":    cp["lon"],
                "source": "loftr",
                "idx":    res["idx"],
            })
    return cps


def deduplicate(cps: list[dict], min_dist_px: float) -> list[dict]:
    """
    중복된 제어점을 제거합니다.
    LoFTR 점이 지오코딩 점보다 우선 (정확도가 높음).
    """
    if not cps:
        return []

    priority = {"loftr": 0, "geocoded": 1}
    ordered  = sorted(cps, key=lambda c: (priority.get(c["source"], 9), c.get("idx", 0)))

    kept: list[dict] = []
    for cp in ordered:
        px, py = cp["img_x"], cp["img_y"]
        if any(math.hypot(px - k["img_x"], py - k["img_y"]) < min_dist_px for k in kept):
            continue
        kept.append(cp)
    return kept


# ---------------------------------------------------------------------------
# TPS / 아핀 적합
# ---------------------------------------------------------------------------

def _build_tps(src: np.ndarray, dst: np.ndarray, smoothing: float):
    """박판 스플라인(TPS) 적합 (scipy ≥ 1.7 필요).

    src: (N, 2) — [img_x, img_y]
    dst: (N, 2) — [lat, lon]
    RBFInterpolator 인스턴스를 반환합니다.
    """
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)


def _build_affine(src: np.ndarray, dst: np.ndarray):
    """
    제어점이 4개 미만일 경우 최소제곱 아핀 적합.
    src: (N, 2), dst: (N, 2)
    쿼리(M, 2)를 입력받아 (M, 2)를 반환하는 호출 가능한 함수를 반환합니다.
    """
    # 증강: [x, y, 1] @ A ≈ [lat, lon]
    A_src = np.column_stack([src, np.ones(len(src))])  # (N, 3)
    A, _, _, _ = np.linalg.lstsq(A_src, dst, rcond=None)  # (3, 2)

    def _predict(query: np.ndarray) -> np.ndarray:
        q_aug = np.column_stack([query, np.ones(len(query))])
        return q_aug @ A

    return _predict


def build_model(cps: list[dict], smoothing: float):
    """제어점 개수에 따라 적절한 공간 모델을 구축합니다."""
    src = np.array([[c["img_x"], c["img_y"]] for c in cps], dtype=np.float64)
    dst = np.array([[c["lat"],   c["lon"]]   for c in cps], dtype=np.float64)

    if len(cps) >= 4:
        try:
            tps = _build_tps(src, dst, smoothing)
            return tps, "tps", src, dst
        except Exception as exc:
            print(f"  [경고] TPS 실패 ({exc}), 아핀으로 대체합니다.")

    affine_fn = _build_affine(src, dst)
    return affine_fn, "affine", src, dst


def self_residuals(model, src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """자기 잔차(훈련 집합 적합 품질)를 도 단위로 계산합니다."""
    try:
        pred = model(src)
        res = np.linalg.norm(pred - dst, axis=1)
        return float(np.mean(res)), float(np.max(res))
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# 범위 진단
# ---------------------------------------------------------------------------

def _coverage(cps: list[dict], image_hw: tuple[int, int]) -> dict:
    """이미지에서 제어점의 공간 범위와 품질을 평가합니다."""
    if len(cps) < 2:
        return {"n_cps": len(cps), "quality": "insufficient", "coverage_pct": 0.0}

    h, w = image_hw
    xs = [c["img_x"] / w for c in cps]
    ys = [c["img_y"] / h for c in cps]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    cov = x_span * y_span * 100.0

    n = len(cps)
    quality = "poor"
    if n >= 4  and cov > 5.0:  quality = "fair"
    if n >= 6  and cov > 15.0: quality = "good"
    if n >= 10 and cov > 30.0: quality = "excellent"

    return {
        "n_cps":        n,
        "quality":      quality,
        "coverage_pct": round(cov, 2),
        "x_span_pct":   round(x_span * 100.0, 2),
        "y_span_pct":   round(y_span * 100.0, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """명령행 인자 파서를 구축합니다."""
    p = argparse.ArgumentParser(
        description="LoFTR 특징점으로 TPS / 아핀 정렬 모델을 구축합니다."
    )
    p.add_argument("--loftr-json", required=True,
                   help="06_loftr_match.py의 LoFTR 매칭 JSON")
    p.add_argument("--smoothing", type=float, default=1e-4,
                   help="TPS 평활화 매개변수 (기본값: 1e-4; 노이즈가 많을 때 증가)")
    p.add_argument("--min-dist-px", type=float, default=10.0,
                   help="중복 제거용 픽셀 거리 임계값 (기본값: 10.0)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/06.tps/",
                   help="출력 디렉토리 (기본값: ./path-to-gpx/output/06.tps/)")
    return p


def main() -> int:
    """메인 함수: LoFTR 특징점으로만 TPS 모델을 구축하고 JSON으로 저장합니다."""
    args = _build_parser().parse_args()

    # LoFTR JSON 파일 확인 (필수)
    loftr_path = Path(args.loftr_json) if args.loftr_json else None
    if not loftr_path or not loftr_path.exists():
        print(f"[오류] LoFTR JSON이 필요합니다: {loftr_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LoFTR 데이터 로드
    loftr_data = json.loads(loftr_path.read_text(encoding="utf-8"))
    image_path = loftr_data.get("image", "unknown")
    stem = Path(image_path).stem if image_path else "output"
    image_width = loftr_data.get("image_width", 512)
    image_height = loftr_data.get("image_height", 512)
    image_hw = (image_height, image_width)

    # LoFTR 기반 제어점만 추출
    all_cps = _loftr_cps(loftr_data)
    print(f"[정보] LoFTR 제어점: {len(all_cps)}")

    # 중복 제거
    all_cps = deduplicate(all_cps, min_dist_px=args.min_dist_px)
    print(f"[정보] 중복 제거 후 제어점: {len(all_cps)}")

    # 범위 평가
    cov = _coverage(all_cps, image_hw)
    print(f"[정보] 공간 범위: {cov}")

    # 최소 제어점 검증
    if len(all_cps) < 2:
        print(
            "[오류] 최소 2개 이상의 제어점이 필요합니다. "
            "06_loftr_match.py에서 더 많은 LoFTR 특징점을 생성하세요."
        )
        return 1

    if len(all_cps) < 4:
        print(f"[경고] {len(all_cps)}개의 제어점만 있음 — 아핀 변환 사용 (TPS는 4개 이상 필요).")

    # 모델 구축 및 자기 잔차 계산
    model, model_type, src_arr, dst_arr = build_model(all_cps, args.smoothing)
    mean_res, max_res = self_residuals(model, src_arr, dst_arr)
    print(f"[정보] 모델_유형={model_type}  자기_잔차 평균={mean_res:.6f}°  최대={max_res:.6f}°")
    if mean_res > 0.05:
        print(
            "  [경고] 높은 자기 잔차 — 특징점이 불일치할 수 있습니다. "
            "LoFTR 매칭 품질을 확인하세요."
        )

    # 결과 JSON 저장
    out_json = output_dir / f"{stem}_tps_model.json"
    out_json.write_text(json.dumps({
        "image":        image_path,
        "image_width":  image_width,
        "image_height": image_height,
        "model_type":   model_type,
        "smoothing":    args.smoothing,
        "n_loftr_cps":  len(all_cps),
        "n_total_cps":  len(all_cps),
        "coverage_quality":            cov,
        "tps_self_residual_mean_deg":  mean_res,
        "tps_self_residual_max_deg":   max_res,
        "control_points": all_cps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[완료] {model_type.upper()} 모델 ({len(all_cps)}개 제어점) → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

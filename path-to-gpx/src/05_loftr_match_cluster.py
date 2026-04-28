'''
05_loftr_match_cluster.py — 군집 특징점 기반 지오레퍼런싱 (Clustered Georeferencing)

[문제]
  05_loftr_match.py가 생성한 제어점 중 일부는 패치 내에서 공간적으로 고립된
  단독 매칭에서 비롯됩니다. 이런 고립 특징점은 오매칭 확률이 높고,
  TPS 변환식에 국소적 왜곡을 유발하여 최종 GPX 경로를 부정확하게 만듭니다.

[해결]
  MAGSAC++ 인라이어 필터링 이후 DBSCAN 군집화를 추가로 적용합니다.
    - 반경 eps_px 내에 min_pts 개 이상의 이웃이 있는 점 → 군집 멤버 (유지)
    - 반경 내 이웃이 부족한 고립 점            → 노이즈  (제거)
  밀집된 군집의 매칭만으로 로컬 픽셀↔GPS 관계식을 구성하므로
  변환 정확도가 향상됩니다.

입력
----
--anchors-json   04_anchor_patches.py 출력 앵커 JSON
--eps-px         DBSCAN 군집 반경 (픽셀, 기본값: 40)
--min-pts        군집 성립 최소 점 수 (기본값: 3)
--confidence     LoFTR 신뢰도 임계값 (기본값: 0.2)
--magsac-thr     MAGSAC++ 재투영 임계값 (픽셀, 기본값: 3.0)
--output-dir     출력 디렉토리

출력
----
<output-dir>/<stem>_cluster_matches.json  — 군집 필터 후 제어점 (06_tps_align.py 호환)
<output-dir>/<stem>_cluster_vis/          — 앵커별 군집 시각화 이미지

지금까지 흐름:
  1. 01_ocr.py            → 텍스트 후보 추출
  2. 02_geo_coding.py     → GPS 좌표 변환
  3. 03_map_tiles.py      → Naver 지도 타일 다운로드
  4. 04_anchor_patches.py → 앵커 패치 추출
  5. 이 스크립트          → 군집 특징점 기반 정밀 제어점 생성
  6. 06_tps_align.py      → TPS/아핀 모델 구축
  7. 07_skeleton_to_gps.py→ 스켈레톤 → GPX 변환
'''
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 지오트랜스폼 헬퍼: 참조 타일 픽셀 → (위도, 경도)
# ---------------------------------------------------------------------------

def tile_px_to_latlon(px: float, py: float, gt: dict) -> tuple[float, float]:
    '''
    Naver Maps Static API 타일의 픽셀 좌표를 위도·경도로 변환한다.

    Web Mercator 역변환 공식 사용:
      - X(경도): 선형 관계 — (px - center_px) * 360 / (256 * 2^zoom)
      - Y(위도): 역 Mercator — 음수 부호(이미지 Y축 반전) + 비선형 공식
        * 선형 근사(÷111320)는 위도에서 cos(lat)만큼 오차 발생 (한국 기준 ~26%)
        * 정확한 역변환: lat = 2·atan(exp(y_merc)) - π/2

    gt 구조 (04_anchor_patches.py 저장 형식):
      zoom, center_lat, center_lon, center_pixel_x, center_pixel_y, patch_size
    '''
    zoom       = gt["zoom"]
    center_lat = gt["center_lat"]
    center_lon = gt["center_lon"]
    center_px  = gt["center_pixel_x"]
    center_py  = gt["center_pixel_y"]

    # 줌 레벨에서 전체 세계 픽셀 너비 (표준 Web Mercator: 256 * 2^zoom)
    total_px = 256.0 * (1 << zoom)

    # 경도: Web Mercator에서 선형 관계
    dlon = (px - center_px) * 360.0 / total_px

    # 위도: 정확한 역 Mercator 공식
    # - 음수 부호: 이미지 Y축(아래가 +)과 위도(북쪽이 +)가 반대 방향
    # - Mercator 공식: y_merc = ln(tan(π/4 + lat/2))
    lat_merc_center = math.log(math.tan(math.pi / 4.0 + math.radians(center_lat) / 2.0))
    dy_merc = -(py - center_py) * 2.0 * math.pi / total_px
    lat = math.degrees(2.0 * math.atan(math.exp(lat_merc_center + dy_merc)) - math.pi / 2.0)

    lon = center_lon + dlon
    return lat, lon


# ---------------------------------------------------------------------------
# LoFTR 특징점 매칭
# ---------------------------------------------------------------------------

def _match_loftr(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    kornia LoFTR로 두 패치를 매칭한다.
    kornia 미설치 시 ImportError 발생 → 호출부에서 SIFT 폴백 처리.

    반환: (mkpts0, mkpts1, conf) — 각각 (N, 2) float32, (N,) float32
    '''
    import torch
    import kornia.feature as KF

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def _to_tensor(bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # LoFTR 입력은 8의 배수 크기만 허용
        h8 = (h // 8) * 8 or 8
        w8 = (w // 8) * 8 or 8
        if (h8, w8) != (h, w):
            gray = cv2.resize(gray, (w8, h8), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t.to(device)

    with torch.inference_mode():
        out = matcher({"image0": _to_tensor(img0_bgr), "image1": _to_tensor(img1_bgr)})

    mkpts0 = out["keypoints0"].cpu().numpy().astype(np.float32)
    mkpts1 = out["keypoints1"].cpu().numpy().astype(np.float32)
    conf   = out["confidence"].cpu().numpy().astype(np.float32)

    keep = conf >= conf_thr
    return mkpts0[keep], mkpts1[keep], conf[keep]


# ---------------------------------------------------------------------------
# SIFT 폴백 매칭
# ---------------------------------------------------------------------------

def _match_sift(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    SIFT + CLAHE + Lowe ratio-test 매칭 (kornia 미설치 시 폴백).

    CLAHE로 국소 대비를 향상시킨 뒤 Canny 엣지 위주로 특징점을 추출하여
    도면 이미지(저대비, 모노톤)에서도 안정적인 매칭을 가능하게 한다.
    '''
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        # 그레이스케일 변환 후 CLAHE로 대비 향상
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        # 엣지 팽창: SIFT 특징점이 엣지 근처에 잘 붙도록
        return cv2.dilate(edges, np.ones((2, 2), np.uint8))

    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
    kp0, des0 = sift.detectAndCompute(_preprocess(img0_bgr), None)
    kp1, des1 = sift.detectAndCompute(_preprocess(img1_bgr), None)

    empty = (np.zeros((0, 2), np.float32),) * 2 + (np.zeros(0, np.float32),)
    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return empty

    # BFMatcher + Lowe ratio-test (0.75)
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des0, des1, k=2)
    pts0, pts1 = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                pts0.append(kp0[m.queryIdx].pt)
                pts1.append(kp1[m.trainIdx].pt)

    if not pts0:
        return empty

    a0 = np.array(pts0, dtype=np.float32)
    a1 = np.array(pts1, dtype=np.float32)
    return a0, a1, np.ones(len(pts0), dtype=np.float32)


# ---------------------------------------------------------------------------
# MAGSAC++ / RANSAC 호모그래피 추정
# ---------------------------------------------------------------------------

# MAGSAC++ 사용 가능 여부를 모듈 로드 시점에 확인
_MAGSAC = getattr(cv2, "USAC_MAGSAC", None)


def _find_homography_robust(
    pts0: np.ndarray,
    pts1: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    '''
    MAGSAC++ (불가 시 RANSAC)으로 호모그래피를 추정하고 인라이어 마스크를 반환한다.

    pts0 / pts1: (N, 2) float32, threshold: 재투영 오차 임계값 (픽셀)
    '''
    if len(pts0) < 4:
        return None, None

    p0 = pts0.reshape(-1, 1, 2).astype(np.float64)
    p1 = pts1.reshape(-1, 1, 2).astype(np.float64)

    if _MAGSAC is not None:
        try:
            H, mask = cv2.findHomography(p0, p1, _MAGSAC, threshold,
                                         confidence=0.999, maxIters=10_000)
            if H is not None:
                return H, mask
        except cv2.error:
            pass  # MAGSAC++ 실패 시 RANSAC으로 폴백

    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, threshold)
    return H, mask


# ---------------------------------------------------------------------------
# ★ 핵심: DBSCAN 군집 필터링
# ---------------------------------------------------------------------------

def filter_clustered_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    eps_px: float,
    min_pts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    MAGSAC++ 인라이어 집합에서 공간적으로 고립된 점을 추가로 제거한다.

    알고리즘
    --------
    1차 시도: sklearn DBSCAN
      - 소스 패치 좌표(pts0)를 기준으로 군집화
      - label == -1 (노이즈) 인 점은 고립 특징점 → 제거
      - label >= 0 인 점만 유지

    폴백 (sklearn 미설치): 반경 밀도 필터
      - 각 점에 대해 eps_px 반경 내 이웃 수를 O(N²) 계산
      - min_pts - 1 개 이상의 이웃을 가진 점만 유지 (자기 자신 제외)

    왜 소스 패치 기준으로 군집화하는가?
      - 소스 패치 좌표 = 마라톤 이미지의 국소 영역
      - 같은 구역의 특징점은 동일한 국소 변환을 공유 → 상호 일관성 높음
      - 참조 타일 기준으로 군집화해도 결과는 유사하나,
        소스 좌표가 최종 img_x/img_y 제어점 기반이므로 소스 기준이 적합

    매개변수
    --------
    pts0    : MAGSAC++ 인라이어 소스 좌표 (N, 2) float32
    pts1    : MAGSAC++ 인라이어 참조 좌표 (N, 2) float32
    eps_px  : 군집 반경 (픽셀)
    min_pts : 군집 최소 구성 점 수

    반환
    ----
    (pts0_kept, pts1_kept, cluster_labels)
      pts0_kept / pts1_kept : 군집 멤버만 남긴 배열 (M ≤ N, 2)
      cluster_labels        : N개 점 각각의 군집 번호 (-1 = 고립/노이즈)
    '''
    n = len(pts0)
    if n == 0:
        empty = np.zeros((0, 2), dtype=pts0.dtype)
        return empty, empty, np.full(0, -1, dtype=np.int32)

    # ── 1차: sklearn DBSCAN ───────────────────────────────────────────────
    try:
        from sklearn.cluster import DBSCAN
        # DBSCAN은 (N, 2) 배열과 eps, min_samples를 받아 label 배열 반환
        # min_samples 는 점 자신을 포함하므로 min_pts 그대로 전달
        labels = DBSCAN(eps=eps_px, min_samples=min_pts).fit_predict(
            pts0.astype(np.float64)
        ).astype(np.int32)
    except ImportError:
        # ── 폴백: 반경 밀도 필터 (O(N²)) ─────────────────────────────────
        # sklearn 없을 때 각 점의 반경 내 이웃 수를 직접 계산
        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0
        visited    = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            # pts0[i]를 중심으로 eps_px 내에 있는 모든 점 탐색
            diffs = pts0 - pts0[i]             # (N, 2)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])  # (N,)
            neighbors = np.where(dists <= eps_px)[0]

            if len(neighbors) < min_pts:
                # 이웃이 부족 → 현재는 노이즈로 분류 (이후 다른 군집에 흡수될 수 있음)
                continue

            # 군집 형성: BFS 방식으로 군집 확장
            labels[neighbors] = cluster_id
            queue = list(neighbors)
            while queue:
                j = queue.pop()
                if visited[j]:
                    continue
                visited[j] = True
                d2 = pts0 - pts0[j]
                nbrs = np.where(np.hypot(d2[:, 0], d2[:, 1]) <= eps_px)[0]
                for k in nbrs:
                    if labels[k] == -1:
                        labels[k] = cluster_id
                    if not visited[k]:
                        queue.append(k)
            cluster_id += 1

    # 군집 멤버 (label >= 0) 인 인덱스만 유지
    keep = labels >= 0
    return pts0[keep], pts1[keep], labels


# ---------------------------------------------------------------------------
# 군집 시각화
# ---------------------------------------------------------------------------

def _draw_cluster_visualization(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    pts0_all: np.ndarray,
    pts1_all: np.ndarray,
    pts0_kept: np.ndarray,
    pts1_kept: np.ndarray,
    cluster_labels_all: np.ndarray,
    output_path: Path,
) -> None:
    '''
    군집 필터링 전/후 매칭 결과를 한 장의 이미지로 비교 시각화한다.

    레이아웃:
      [왼쪽] 필터링 전 — 인라이어 전체
               회색 = 고립(제거될) 매칭, 색상 = 군집 멤버
      [오른쪽] 필터링 후 — 군집 멤버만
               각 군집을 다른 색상으로 표시

    이미지 양쪽에 소스 패치(img0)와 참조 타일(img1)을 배치하고
    점과 연결선으로 매칭을 표시한다.
    '''
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    # 고유 군집 수 및 색상 팔레트 준비
    unique_ids = sorted(set(cluster_labels_all[cluster_labels_all >= 0]))
    # 군집별 고유 색상: HSV 색상환에서 균등 분할
    def _cluster_color(cid: int) -> tuple:
        if len(unique_ids) == 0:
            return (0, 200, 200)
        hue = int((unique_ids.index(cid) / max(1, len(unique_ids))) * 180)
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr))

    margin = 30
    header_h = 70
    panel_h  = max(h0, h1) + margin * 2 + header_h
    panel_w  = w0 + w1 + margin * 3
    total_w  = panel_w * 2 + margin

    vis = np.full((panel_h, total_w, 3), 245, dtype=np.uint8)

    for panel_idx, (pts0_show, pts1_show, title, use_clusters) in enumerate([
        (pts0_all,   pts1_all,   "Before: All Inliers (gray=isolated)", True),
        (pts0_kept,  pts1_kept,  "After:  Clustered Only",              False),
    ]):
        ox = panel_idx * (panel_w + margin)  # 패널 X 오프셋

        # 헤더 배경 및 제목
        cv2.rectangle(vis, (ox, 0), (ox + panel_w, header_h), (220, 220, 220), -1)
        cv2.rectangle(vis, (ox, 0), (ox + panel_w, header_h), (80, 80, 80), 1)
        cv2.putText(vis, title, (ox + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        # 통계 문자열
        n_before = len(pts0_all)
        n_after  = len(pts0_kept)
        pct = f"{n_after/max(1, n_before)*100:.1f}%" if n_before else "0%"
        stat_str = (f"Inliers: {n_before}  Kept: {n_after} ({pct})"
                    if panel_idx == 0
                    else f"Clustered matches: {n_after}")
        cv2.putText(vis, stat_str, (ox + 10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

        iy = header_h + margin  # 이미지 Y 시작
        # 소스 패치(img0) 붙이기
        vis[iy:iy+h0, ox+margin:ox+margin+w0] = img0_bgr
        # 참조 타일(img1) 붙이기
        ix1 = ox + margin + w0 + margin
        vis[iy:iy+h1, ix1:ix1+w1] = img1_bgr

        # 매칭 선과 점 그리기
        for k in range(len(pts0_show)):
            px0 = (int(pts0_show[k, 0]) + ox + margin, int(pts0_show[k, 1]) + iy)
            px1 = (int(pts1_show[k, 0]) + ix1,          int(pts1_show[k, 1]) + iy)

            if use_clusters:
                # "Before" 패널: 군집 멤버는 군집 색상, 고립 점은 회색
                # pts0_show == pts0_all이므로 cluster_labels_all을 직접 참조
                lbl = cluster_labels_all[k]
                if lbl >= 0:
                    color = _cluster_color(lbl)
                    thickness, r = 1, 4
                else:
                    color = (160, 160, 160)   # 회색 = 고립 (제거 대상)
                    thickness, r = 1, 3
            else:
                # "After" 패널: 남은 점을 색상 그라데이션으로 표시
                hue = int(k / max(1, len(pts0_show)) * 160)
                hsv = np.uint8([[[hue, 200, 210]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(map(int, bgr))
                thickness, r = 2, 5

            cv2.circle(vis, px0, r, color, -1)
            cv2.circle(vis, px1, r, color, -1)
            cv2.line(vis, px0, px1, color, thickness, cv2.LINE_AA)

    # 범례
    legend_y = panel_h - 20
    cv2.putText(vis, "Gray = isolated (removed)  |  Color = cluster member (kept)",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

    cv2.imwrite(str(output_path), vis)


# ---------------------------------------------------------------------------
# 앵커 1개 처리
# ---------------------------------------------------------------------------

def _process_anchor(
    anchor: dict,
    conf_thr: float,
    magsac_thr: float,
    eps_px: float,
    min_pts: int,
    vis_dir: Path | None,
) -> dict[str, Any]:
    '''
    앵커 1쌍(소스 패치 + 참조 타일)에 대해 아래 단계를 순차 실행한다:

      1. LoFTR / SIFT 특징점 매칭
      2. MAGSAC++ / RANSAC 인라이어 필터 (기하학적 이상치 제거)
      3. DBSCAN 군집 필터 (고립 단독 매칭 추가 제거)
      4. 군집 매칭 좌표를 소스 이미지 전체 좌표 및 GPS 좌표로 변환
      5. (선택) 시각화 이미지 저장

    반환 딕셔너리는 06_tps_align.py의 _loftr_cps() 파싱 형식과 호환된다.
    '''
    idx      = anchor["idx"]
    src_path = Path(anchor["src_patch"])
    ref_path = Path(anchor["ref_tile"])
    gt       = anchor["geotransform"]   # 참조 타일 지오트랜스폼
    off      = anchor["src_offset"]     # 소스 패치→전체 이미지 오프셋

    base = {"idx": idx, "text": anchor.get("text", ""), "control_points": []}

    # 파일 존재 확인
    for p, label in [(src_path, "src_patch"), (ref_path, "ref_tile")]:
        if not p.exists():
            return {**base, "status": f"missing_{label}"}

    img0 = cv2.imread(str(src_path))   # 소스 패치 (마라톤 이미지 크롭)
    img1 = cv2.imread(str(ref_path))   # 참조 타일 (Naver Maps)
    if img0 is None or img1 is None:
        return {**base, "status": "load_error"}

    # ── 단계 1: 특징점 매칭 ────────────────────────────────────────────────
    matcher_used = "loftr"
    try:
        mkpts0, mkpts1, conf = _match_loftr(img0, img1, conf_thr)
    except ImportError:
        print("    [WARN] kornia 미설치 → SIFT 폴백")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"
    except Exception as exc:
        print(f"    [WARN] LoFTR 오류 ({exc}) → SIFT 폴백")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"

    raw_n = len(mkpts0)
    print(f"    원본 매칭={raw_n}  매처={matcher_used}")

    if raw_n < 4:
        return {**base, "status": "too_few_matches",
                "raw_matches": raw_n, "matcher_used": matcher_used}

    # ── 단계 2: MAGSAC++ / RANSAC 인라이어 필터 ───────────────────────────
    H, mask = _find_homography_robust(mkpts0, mkpts1, magsac_thr)
    inlier_n = int(mask.ravel().astype(bool).sum()) if mask is not None else 0
    print(f"    MAGSAC++ 인라이어={inlier_n}/{raw_n}")

    if H is None or inlier_n < 4:
        return {**base, "status": "homography_failed",
                "raw_matches": raw_n, "matcher_used": matcher_used,
                "inlier_matches": inlier_n}

    inlier_mask = mask.ravel().astype(bool)
    pts0_in = mkpts0[inlier_mask]   # MAGSAC++ 인라이어 소스 좌표
    pts1_in = mkpts1[inlier_mask]   # MAGSAC++ 인라이어 참조 좌표

    # ── 단계 3: DBSCAN 군집 필터 ──────────────────────────────────────────
    pts0_kept, pts1_kept, cluster_labels = filter_clustered_matches(
        pts0_in, pts1_in, eps_px=eps_px, min_pts=min_pts
    )
    cluster_n = len(pts0_kept)
    removed_n = inlier_n - cluster_n
    n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
    print(f"    DBSCAN 군집={n_clusters}개  유지={cluster_n}  제거(고립)={removed_n}")

    if cluster_n < 2:
        return {**base, "status": "no_cluster",
                "raw_matches": raw_n, "matcher_used": matcher_used,
                "inlier_matches": inlier_n, "cluster_matches": cluster_n}

    # ── 단계 4: 군집 매칭 → GPS 제어점 변환 ──────────────────────────────
    # src_offset: 소스 패치가 전체 마라톤 이미지에서 잘린 위치 오프셋
    off_x = float(off["offset_x"])
    off_y = float(off["offset_y"])

    control_points: list[dict] = []
    for (px_p, py_p), (tx_t, ty_t) in zip(pts0_kept, pts1_kept):
        # 참조 타일 픽셀 → (위도, 경도) 변환
        lat, lon = tile_px_to_latlon(float(tx_t), float(ty_t), gt)
        control_points.append({
            # 마라톤 전체 이미지 좌표 (TPS 모델이 사용하는 좌표계)
            "img_x":   float(px_p) + off_x,
            "img_y":   float(py_p) + off_y,
            "lat":     lat,
            "lon":     lon,
            # 디버깅 목적의 로컬 좌표 (패치 내 좌표, 타일 픽셀 좌표)
            "patch_x": float(px_p),
            "patch_y": float(py_p),
            "tile_x":  float(tx_t),
            "tile_y":  float(ty_t),
        })

    # ── 단계 5: 시각화 저장 ────────────────────────────────────────────────
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"anchor_{idx:03d}_cluster_filter.jpg"
        _draw_cluster_visualization(
            img0_bgr=img0,
            img1_bgr=img1,
            pts0_all=pts0_in,            # MAGSAC++ 인라이어 전체
            pts1_all=pts1_in,
            pts0_kept=pts0_kept,         # 군집 필터 후 남은 점
            pts1_kept=pts1_kept,
            cluster_labels_all=cluster_labels,
            output_path=vis_path,
        )
        print(f"    시각화 저장 → {vis_path.name}")

    return {
        **base,
        "status":           "ok",
        "matcher_used":     matcher_used,
        "raw_matches":      raw_n,
        "inlier_matches":   inlier_n,
        "cluster_matches":  cluster_n,
        "n_clusters":       n_clusters,
        "removed_isolated": removed_n,
        "inlier_ratio":     inlier_n / raw_n,
        "cluster_ratio":    cluster_n / inlier_n if inlier_n else 0.0,
        "control_points":   control_points,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="군집 특징점 기반 지오레퍼런싱 — 고립 매칭 제거 후 정밀 제어점 생성"
    )
    parser.add_argument(
        "--anchors-json", required=True,
        help="04_anchor_patches.py 출력 앵커 JSON 경로",
    )
    parser.add_argument(
        "--eps-px", type=float, default=40.0,
        help="DBSCAN 군집 반경 (픽셀, 기본값: 40.0)",
    )
    parser.add_argument(
        "--min-pts", type=int, default=3,
        help="군집 성립 최소 점 수 (기본값: 3, 이 미만은 고립으로 분류)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.2,
        help="LoFTR 신뢰도 임계값 (기본값: 0.2)",
    )
    parser.add_argument(
        "--magsac-thr", type=float, default=3.0,
        help="MAGSAC++ 재투영 임계값 (픽셀, 기본값: 3.0)",
    )
    parser.add_argument(
        "--output-dir", default="./path-to-gpx/output/05.loftr_cluster/",
        help="출력 디렉토리 (기본값: ./path-to-gpx/output/05.loftr_cluster/)",
    )
    parser.add_argument(
        "--no-vis", action="store_true", default=False,
        help="시각화 이미지 저장 생략 (속도 우선 시 사용)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    anchors_path = Path(args.anchors_json)
    if not anchors_path.exists() or not anchors_path.is_file():
        raise FileNotFoundError(f"--anchors-json 파일을 찾을 수 없음: {anchors_path}")

    eps_px  = float(args.eps_px)
    min_pts = int(args.min_pts)
    if min_pts < 2:
        raise ValueError("--min-pts는 2 이상이어야 합니다.")
    if eps_px <= 0:
        raise ValueError("--eps-px는 양수여야 합니다.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 앵커 JSON 로드
    data    = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = data["anchors"]
    stem    = Path(data["image"]).stem

    print(f"[INFO] 앵커 수: {len(anchors)}  이미지: {stem}")
    print(f"[INFO] DBSCAN 파라미터: eps={eps_px}px, min_pts={min_pts}")
    print(f"[INFO] MAGSAC++ 사용: {'예' if _MAGSAC is not None else '아니오 (RANSAC 대체)'}")

    # 시각화 디렉토리 준비
    vis_dir: Path | None = None
    if not args.no_vis:
        vis_dir = output_dir / f"{stem}_cluster_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 시각화 디렉토리: {vis_dir}")

    # 앵커별 처리
    results: list[dict] = []
    total_cps      = 0
    total_removed  = 0

    for i, anchor in enumerate(anchors):
        print(f"\n[{i+1}/{len(anchors)}] 앵커 idx={anchor['idx']}  '{anchor.get('text', '')}'")
        res = _process_anchor(
            anchor,
            conf_thr=args.confidence,
            magsac_thr=args.magsac_thr,
            eps_px=eps_px,
            min_pts=min_pts,
            vis_dir=vis_dir,
        )
        results.append(res)
        n_cps = len(res.get("control_points", []))
        total_cps     += n_cps
        total_removed += res.get("removed_isolated", 0)
        print(f"    → 상태={res['status']}  제어점={n_cps}")

    # 결과 JSON 저장 (06_tps_align.py 호환 형식)
    # 06_tps_align.py의 --loftr-json 인자에 이 파일을 전달하면 됨
    from datetime import datetime, timezone
    out_json = output_dir / f"{stem}_cluster_matches.json"
    out_json.write_text(json.dumps({
        "created_at_utc":    datetime.now(timezone.utc).isoformat(),
        "anchors_json":      str(anchors_path),
        "image":             data["image"],
        "image_width":       data["image_width"],
        "image_height":      data["image_height"],
        "confidence_threshold": args.confidence,
        "magsac_threshold":  args.magsac_thr,
        "eps_px":            eps_px,
        "min_pts":           min_pts,
        "total_control_points": total_cps,
        "total_removed_isolated": total_removed,
        # 06_tps_align.py의 _loftr_cps()가 읽는 키
        "anchor_results":    results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_n    = sum(1 for r in results if r.get("status") == "ok")
    fail_n  = len(results) - ok_n
    print(
        f"\n[완료] 앵커 {len(anchors)}개 처리 (성공={ok_n}, 실패={fail_n})"
        f"\n       제어점 총 {total_cps}개, 고립 제거 {total_removed}개"
        f"\n       출력 → {out_json}"
    )

    # 제어점이 하나도 없으면 비정상 종료 코드 반환
    return 0 if total_cps > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

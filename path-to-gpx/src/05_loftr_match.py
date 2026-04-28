"""06_loftr_match.py - 교차 도메인 특징점 매칭 (LoFTR + MAGSAC++)

05_anchor_patches.py에서 생성한 각 소스 패치/참조 타일 쌍에 대해:
  1. LoFTR (kornia)을 실행하여 조밀한 교차 도메인 대응 생성
     kornia가 설치되지 않은 경우 SIFT + ratio-test로 대체
  2. MAGSAC++ (cv2.USAC_MAGSAC, RANSAC으로 대체 가능)을 적용하여
     이상치 제거 및 각 앵커에 대한 로컬 호모그래피 추정
  3. 인라이어 타일-픽셀 대응을 05_anchor_patches.py에서 저장한
     지오트랜스폼을 통해 (위도, 경도)로 변환

출력은 Image-A 픽셀 ↔ (위도, 경도) 제어점의 목록으로,
각 앵커당 최대 100개 매칭이 포함됨 (신뢰도 임계값과 MAGSAC++ 결과에 따라 다름).

입력
------
--anchors-json   05_anchor_patches.py의 출력
--confidence     LoFTR 신뢰도 임계값 (기본값: 0.2)
--magsac-thr     MAGSAC++ 재투영 임계값 (픽셀 단위, 기본값: 3.0)
--output-dir     출력 디렉토리 (기본값: ./path-to-gpx/output/loftr/)

출력
------
<output-dir>/<stem>_loftr_matches.json
<output-dir>/<stem>_match_visualizations/ (매칭 시각화 이미지)

지금까지 흐름:
1. 01_ocr.py: 마라톤 코스 사진에서 텍스트 감지 및 인식 → 장소 이름 후보 추출
2. 02_geo_coding.py: 장소 이름 후보를 Kakao 지오코딩 API로 변환 → (위도, 경도) 후보 생성
3. 03_map_tiles.py: 각 (위도, 경도) 후보에 대해 Naver Maps Static API로 타일 이미지 다운로드 → 타일별 지오트랜스폼 정보 저장
4. 04_anchor_patches.py: 각 타일에서 OCR 감지된 텍스트 주변에 앵커 패치 생성 → 앵커별 소스 패치와 참조 타일 패치 저장
5. 05_loftr_match.py (현재 스크립트): 각 앵커 패치 쌍에 대해 LoFTR 매칭 → MAGSAC++로 이상치 제거 → 매칭 결과를 (위도, 경도)로 변환하여 저장 + 시각화 이미지 생성
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 지오트랜스폼 헬퍼 함수들
# ---------------------------------------------------------------------------

def _tile_float_to_latlon(tx: float, ty: float, zoom: int) -> tuple[float, float]:
    """타일 좌표를 위도, 경도로 변환"""
    n = 1 << zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lat, lon


def tile_px_to_latlon(px: float, py: float, gt: dict) -> tuple[float, float]:
    """
    참조 타일 이미지의 픽셀 (px, py)을 (위도, 경도)로 변환

    Web Mercator 역변환 공식 사용:
      - X(경도): 선형 관계 — (px - center_px) * 360 / (256 * 2^zoom)
      - Y(위도): 역 Mercator — 음수 부호(이미지 Y축 반전) + 비선형 공식

    gt 구조:
    {
        "zoom": int,
        "center_lat": float,
        "center_lon": float,
        "center_pixel_x": float,  # 타일 내 중심 픽셀 X 좌표
        "center_pixel_y": float,  # 타일 내 중심 픽셀 Y 좌표
        "patch_size": int
    }
    """
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
# LoFTR 특징점 매칭 함수
# ---------------------------------------------------------------------------

def _match_loftr(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    img0 (소스 패치)과 img1 (Map 참조 타일)을 kornia LoFTR로 매칭

    반환: (mkpts0, mkpts1, conf) - shape (N, 2)와 (N,)의 float32 배열
    kornia가 설치되지 않으면 ImportError 발생
    """
    import torch
    import kornia.feature as KF

    device = "cuda" if torch.cuda.is_available() else "cpu"

    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def _to_tensor(bgr: np.ndarray) -> "torch.Tensor":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # LoFTR은 8의 배수 크기 필요
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
# SIFT 폴백 매칭 함수
# ---------------------------------------------------------------------------

def _match_sift(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIFT + CLAHE-엣지 + BF k-NN + Lowe ratio-test 대체 방법"""

    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        return cv2.dilate(edges, np.ones((2, 2), np.uint8))

    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
    kp0, des0 = sift.detectAndCompute(_preprocess(img0_bgr), None)
    kp1, des1 = sift.detectAndCompute(_preprocess(img1_bgr), None)

    empty = (np.zeros((0, 2), np.float32),) * 2 + (np.zeros(0, np.float32),)
    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return empty

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
# 로버스트 호모그래피 추정 (MAGSAC++ / RANSAC)
# ---------------------------------------------------------------------------

# MAGSAC++ 사용 가능 여부 확인
_MAGSAC = getattr(cv2, "USAC_MAGSAC", None)


def _find_homography_robust(
    pts0: np.ndarray,
    pts1: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """MAGSAC++ (또는 불가능시 RANSAC)으로 호모그래피 추정"""
    if len(pts0) < 4:
        return None, None

    p0 = pts0.reshape(-1, 1, 2).astype(np.float64)
    p1 = pts1.reshape(-1, 1, 2).astype(np.float64)

    if _MAGSAC is not None:
        try:
            H, mask = cv2.findHomography(
                p0, p1, _MAGSAC, threshold, confidence=0.999, maxIters=10_000
            )
            if H is not None:
                return H, mask
        except cv2.error:
            pass  # MAGSAC++ 실패 시 RANSAC으로 폴백

    # RANSAC 대체 방법
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, threshold)
    return H, mask


# ---------------------------------------------------------------------------
# 시각화 헬퍼 함수들
# ---------------------------------------------------------------------------

def _get_color_for_match(idx: int, total: int) -> tuple:
    """매칭 번호에 따라 색상을 동적으로 생성 (HSV 색상환 이용)"""
    hue = int((idx / max(1, total)) * 180)  # 0-180 범위 (OpenCV HSV)
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, color_bgr))


def _visualize_matches_single(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    모든 특징점 매칭을 한 장의 이미지에 순례대로 표시
    - 좌측: 소스 이미지, 우측: 참조 이미지
    - 각 매칭을 번호와 색상으로 표시
    - 인라이어 vs 아웃라이어 구분
    - 정렬된 순서로 매칭 표시
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    # 여백 및 헤더 설정
    margin = 50
    header_h = 80
    total_h = max(h0, h1) + margin * 2 + header_h
    total_w = w0 + w1 + margin * 3
    
    vis = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    vis[header_h+margin:header_h+margin+h0, margin:margin+w0] = img0_bgr
    vis[header_h+margin:header_h+margin+h1, margin+w0+margin:margin+w0+margin+w1] = img1_bgr

    # 통계 계산
    total_matches = len(mkpts0)
    inlier_count = int(np.sum(inlier_mask))

    # 헤더 영역 그리기
    header_color = (240, 240, 240)
    cv2.rectangle(vis, (0, 0), (total_w, header_h), header_color, -1)
    cv2.rectangle(vis, (0, 0), (total_w, header_h), (0, 0, 0), 2)
    
    title = f"Feature Matching Visualization"
    cv2.putText(vis, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    stats_text = f"Total: {total_matches}  |  Inliers: {inlier_count}  |  Outliers: {total_matches - inlier_count}"
    cv2.putText(vis, stats_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    # 모든 매칭 표시
    for idx in range(total_matches):
        pt0 = tuple(mkpts0[idx].astype(int))
        pt1 = tuple(mkpts1[idx].astype(int))

        is_inlier = inlier_mask[idx] if idx < len(inlier_mask) else False
        
        # 인라이어/아웃라이어에 따라 색상 및 스타일 결정
        if is_inlier:
            color = _get_color_for_match(idx, total_matches)
            line_thickness = 2
            circle_radius = 6
            circle_thickness = 2
        else:
            color = (150, 150, 150)  # 아웃라이어는 회색
            line_thickness = 1
            circle_radius = 4
            circle_thickness = 1

        # 소스 이미지의 특징점 원 그리기
        pt0_offset = (pt0[0] + margin, pt0[1] + header_h + margin)
        cv2.circle(vis, pt0_offset, circle_radius, color, -1)
        cv2.circle(vis, pt0_offset, circle_radius + 2, color, circle_thickness)

        # 참조 이미지의 특징점 원 그리기
        pt1_offset = (pt1[0] + margin + w0 + margin, pt1[1] + header_h + margin)
        cv2.circle(vis, pt1_offset, circle_radius, color, -1)
        cv2.circle(vis, pt1_offset, circle_radius + 2, color, circle_thickness)

        # 두 특징점 연결선 그리기
        cv2.line(vis, pt0_offset, pt1_offset, color, line_thickness)

        # 매칭 번호 표시
        text = f"{idx+1}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = max(margin, pt0_offset[0] - text_size[0] // 2)
        text_y = max(header_h + 20, pt0_offset[1] - 12)
        cv2.putText(vis, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 범례 표시
    legend_y = total_h - 25
    cv2.putText(vis, "Inliers: Colored", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    cv2.putText(vis, "Outliers: Gray", (250, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

    cv2.imwrite(str(output_path), vis)


def _visualize_matches_grid(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
    max_matches: int = 20,
) -> None:
    """
    각 매칭 쌍을 그리드로 배열하여 한 장의 이미지에 표시
    각 셀은 매칭 부분을 확대하여 보여줌
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    
    total_matches = len(mkpts0)
    display_count = min(total_matches, max_matches)
    
    # 그리드 레이아웃 계산
    grid_cols = 4
    grid_rows = (display_count + grid_cols - 1) // grid_cols
    
    patch_size = 120
    padding = 10
    cell_width = patch_size * 2 + padding * 3
    cell_height = patch_size + padding * 2
    
    header_h = 60
    total_w = grid_cols * cell_width + padding * 2
    total_h = header_h + grid_rows * cell_height + padding * 2
    
    vis = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    # 헤더 영역 그리기
    cv2.rectangle(vis, (0, 0), (total_w, header_h), (240, 240, 240), -1)
    cv2.rectangle(vis, (0, 0), (total_w, header_h), (0, 0, 0), 2)
    
    title = f"Feature Matching Grid View"
    cv2.putText(vis, title, (padding, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    stats = f"Showing {display_count}/{total_matches} matches (Inliers: {int(np.sum(inlier_mask[:display_count]))})"
    cv2.putText(vis, stats, (padding, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    # 각 매칭을 그리드 셀에 표시
    for match_idx in range(display_count):
        row = match_idx // grid_cols
        col = match_idx % grid_cols
        
        cell_x = padding + col * cell_width
        cell_y = header_h + padding + row * cell_height
        
        is_inlier = inlier_mask[match_idx] if match_idx < len(inlier_mask) else False
        color = _get_color_for_match(match_idx, total_matches) if is_inlier else (180, 180, 180)
        border_color = (0, 255, 0) if is_inlier else (0, 0, 255)
        
        # 셀 배경 및 테두리 그리기
        cv2.rectangle(vis, (cell_x, cell_y), (cell_x + cell_width, cell_y + cell_height), 
                     (245, 245, 245), -1)
        cv2.rectangle(vis, (cell_x, cell_y), (cell_x + cell_width, cell_y + cell_height), 
                     border_color, 2)
        
        # 소스 이미지 패치 추출 및 표시
        pt0 = mkpts0[match_idx].astype(int)
        x0_start = max(0, pt0[0] - patch_size // 2)
        y0_start = max(0, pt0[1] - patch_size // 2)
        x0_end = min(w0, x0_start + patch_size)
        y0_end = min(h0, y0_start + patch_size)
        patch0 = img0_bgr[y0_start:y0_end, x0_start:x0_end]
        
        patch0_h, patch0_w = patch0.shape[:2]
        vis_y0 = cell_y + padding
        vis_x0 = cell_x + padding
        vis[vis_y0:vis_y0+patch0_h, vis_x0:vis_x0+patch0_w] = patch0
        
        # 참조 이미지 패치 추출 및 표시
        pt1 = mkpts1[match_idx].astype(int)
        x1_start = max(0, pt1[0] - patch_size // 2)
        y1_start = max(0, pt1[1] - patch_size // 2)
        x1_end = min(w1, x1_start + patch_size)
        y1_end = min(h1, y1_start + patch_size)
        patch1 = img1_bgr[y1_start:y1_end, x1_start:x1_end]
        
        patch1_h, patch1_w = patch1.shape[:2]
        vis_y1 = cell_y + padding
        vis_x1 = cell_x + padding + patch_size + padding
        vis[vis_y1:vis_y1+patch1_h, vis_x1:vis_x1+patch1_w] = patch1
        
        # 매칭 번호 표시
        text = f"#{match_idx+1}"
        cv2.putText(vis, text, (cell_x + padding, cell_y + padding - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(str(output_path), vis)


def _visualize_matches_sequential_combined(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    모든 매칭을 누적으로 표시하는 단일 이미지 생성
    왼쪽: 처음 매칭, 오른쪽: 마지막 매칭이 누적되는 방식
    (큰 이미지를 좌우로 배열)
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    
    total_matches = len(mkpts0)
    
    # 각 컬럼(좌우)의 여백 및 헤더 높이 설정
    margin = 40
    header_h = 50
    
    # 전체 시각화 이미지 크기 계산
    cell_h = max(h0, h1) + margin * 2 + header_h
    cell_w = w0 + w1 + margin * 3
    total_w = cell_w * 2 + margin * 3
    
    vis = np.ones((cell_h, total_w, 3), dtype=np.uint8) * 255
    
    # 좌우 2개 컬럼 렌더: (첫 매칭 1개) vs (모든 매칭)
    for col_idx, (num_matches_to_show, label) in enumerate([(1, "First Match"), (total_matches, "All Matches")]):
        col_x = margin + col_idx * (cell_w + margin)
        
        # 각 컬럼용 이미지 생성
        col_vis = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255
        
        # 헤더 영역 그리기
        cv2.rectangle(col_vis, (0, 0), (cell_w, header_h), (240, 240, 240), -1)
        cv2.rectangle(col_vis, (0, 0), (cell_w, header_h), (0, 0, 0), 1)
        cv2.putText(col_vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 이미지 배치
        img_y = header_h + margin
        col_vis[img_y:img_y+h0, margin:margin+w0] = img0_bgr
        col_vis[img_y:img_y+h1, margin+w0+margin:margin+w0+margin+w1] = img1_bgr
        
        # 매칭 그리기 (0부터 num_matches_to_show까지)
        for idx in range(num_matches_to_show):
            pt0 = tuple(mkpts0[idx].astype(int))
            pt1 = tuple(mkpts1[idx].astype(int))
            
            is_inlier = inlier_mask[idx] if idx < len(inlier_mask) else False
            color = _get_color_for_match(idx, total_matches) if is_inlier else (180, 180, 180)
            
            # 점과 연결선 그리기
            pt0_offset = (pt0[0] + margin, pt0[1] + img_y)
            pt1_offset = (pt1[0] + margin + w0 + margin, pt1[1] + img_y)
            
            circle_r = 5 if is_inlier else 3
            thick = 2 if is_inlier else 1
            
            cv2.circle(col_vis, pt0_offset, circle_r, color, -1)
            cv2.circle(col_vis, pt1_offset, circle_r, color, -1)
            cv2.line(col_vis, pt0_offset, pt1_offset, color, thick)
            
            if num_matches_to_show <= 10:  # 매칭이 적을 때만 번호 표시
                cv2.putText(col_vis, str(idx+1), (pt0_offset[0]-5, pt0_offset[1]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 메인 이미지에 복사
        vis[:, col_x:col_x+cell_w] = col_vis
    
    cv2.imwrite(str(output_path), vis)


def _visualize_matches(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
    max_matches_per_image: int = 50,
) -> None:
    """
    특징점 매칭을 시각화하여 이미지로 저장
    각 매칭 쌍을 다른 색상으로 표시하며, 인라이어는 초록색, 아웃라이어는 빨간색으로 표시
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    # 처리
    vis = np.ones((max(h0, h1), w0 + w1, 3), dtype=np.uint8) * 255
    vis[:h0, :w0] = img0_bgr
    vis[:h1, w0:w0+w1] = img1_bgr

    # 처리
    total_matches = len(mkpts0)
    step = max(1, total_matches // max_matches_per_image)

    for idx in range(0, total_matches, step):
        pt0 = tuple(mkpts0[idx].astype(int))
        pt1 = tuple(mkpts1[idx].astype(int))

        is_inlier = inlier_mask[idx] if idx < len(inlier_mask) else False
        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        # 처리
        cv2.circle(vis, pt0, 4, color, -1)
        cv2.circle(vis, pt0, 6, color, 2)

        # 처리
        pt1_offset = (pt1[0] + w0, pt1[1])
        cv2.circle(vis, pt1_offset, 4, color, -1)
        cv2.circle(vis, pt1_offset, 6, color, 2)

        # 처리
        cv2.line(vis, pt0, pt1_offset, color, 1)

        # 처리
        cv2.putText(vis, str(idx // step + 1), (pt0[0] - 10, pt0[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(str(output_path), vis)


def _visualize_matches_sequential(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    output_dir: Path,
    anchor_idx: int,
) -> None:
    """
    각 매칭 쌍을 례로 표시한 시각화 이미지 생성
    각 이미지는 지금까지의 모든 매칭을 누적으로 표시
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    total_matches = len(mkpts0)

    # 처리
    indices_to_save = list(range(min(10, total_matches)))
    indices_to_save.extend(range(10, total_matches, 5))
    if total_matches - 1 not in indices_to_save:
        indices_to_save.append(total_matches - 1)
    indices_to_save = sorted(set(indices_to_save))

    for save_idx in indices_to_save:
        vis = np.ones((max(h0, h1), w0 + w1, 3), dtype=np.uint8) * 255
        vis[:h0, :w0] = img0_bgr
        vis[:h1, w0:w0+w1] = img1_bgr

        # 처리
        for idx in range(save_idx + 1):
            pt0 = tuple(mkpts0[idx].astype(int))
            pt1 = tuple(mkpts1[idx].astype(int))

            is_inlier = inlier_mask[idx] if idx < len(inlier_mask) else False
            color = (0, 255, 0) if is_inlier else (0, 0, 255)

            cv2.circle(vis, pt0, 3, color, -1)
            cv2.circle(vis, pt0, 5, color, 1)

            pt1_offset = (pt1[0] + w0, pt1[1])
            cv2.circle(vis, pt1_offset, 3, color, -1)
            cv2.circle(vis, pt1_offset, 5, color, 1)

            cv2.line(vis, pt0, pt1_offset, color, 1)

        # 처리
        text = f"Anchor {anchor_idx} | Matches: {save_idx + 1}/{total_matches}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        output_file = output_dir / f"anchor_{anchor_idx:03d}_matches_{save_idx+1:04d}.jpg"
        cv2.imwrite(str(output_file), vis)


# ---------------------------------------------------------------------------
# 처리
# ---------------------------------------------------------------------------

def _process_anchor(
    anchor: dict,
    conf_thr: float,
    magsac_thr: float,
    visualization_dir: Path | None = None,
) -> dict[str, Any]:
    idx = anchor["idx"]
    src_path = Path(anchor["src_patch"])
    ref_path = Path(anchor["ref_tile"])
    gt = anchor["geotransform"]
    off = anchor["src_offset"]

    base = {"idx": idx, "text": anchor.get("text", ""), "control_points": []}

    for p, label in [(src_path, "src_patch"), (ref_path, "ref_tile")]:
        if not p.exists():
            return {**base, "status": f"missing_{label}"}

    img0 = cv2.imread(str(src_path))
    img1 = cv2.imread(str(ref_path))
    if img0 is None or img1 is None:
        return {**base, "status": "load_error"}

    # 처리
    matcher_used = "loftr"
    try:
        mkpts0, mkpts1, conf = _match_loftr(img0, img1, conf_thr)
    except ImportError:
        print("    [WARNING] kornia 미설치 - SIFT로 대체합니다.")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"
    except Exception as exc:
        print(f"    [WARNING] LoFTR 오류 ({exc}) - SIFT로 대체합니다.")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"

    raw_n = len(mkpts0)
    print(f"    원본 매칭={raw_n}  매처={matcher_used}")

    if raw_n < 4:
        return {**base, "status": "too_few_matches",
                "raw_matches": raw_n, "matcher_used": matcher_used}

    # ---- MAGSAC++ / RANSAC -----------------------------------------------
    H, mask = _find_homography_robust(mkpts0, mkpts1, magsac_thr)
    inlier_n = int(mask.ravel().astype(bool).sum()) if mask is not None else 0
    print(f"    인라이어={inlier_n}/{raw_n}  호모그래피={'찾음' if H is not None else '실패'}")

    if H is None or inlier_n < 4:
        return {**base, "status": "homography_failed",
                "raw_matches": raw_n, "matcher_used": matcher_used,
                "inlier_matches": inlier_n}

    # 처리
    inlier_mask = mask.ravel().astype(bool)
    pts0_in = mkpts0[inlier_mask]
    pts1_in = mkpts1[inlier_mask]

    off_x = float(off["offset_x"])
    off_y = float(off["offset_y"])

    control_points: list[dict] = []
    for (px_p, py_p), (tx_t, ty_t) in zip(pts0_in, pts1_in):
        lat, lon = tile_px_to_latlon(float(tx_t), float(ty_t), gt)
        control_points.append({
            "img_x":   float(px_p) + off_x,
            "img_y":   float(py_p) + off_y,
            "lat":     lat,
            "lon":     lon,
            "patch_x": float(px_p),
            "patch_y": float(py_p),
            "tile_x":  float(tx_t),
            "tile_y":  float(ty_t),
        })

    # 처리
    if visualization_dir is not None:
        vis_dir = visualization_dir / f"anchor_{idx:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 처리
        all_mask = np.zeros(len(mkpts0), dtype=bool)
        all_mask[inlier_mask] = True
        inlier_only_mask = np.ones(len(pts0_in), dtype=bool)

        # 처리
        vis_output_single = vis_dir / "01_all_inliers_single.jpg"
        _visualize_matches_single(img0, img1, pts0_in, pts1_in, inlier_only_mask, vis_output_single)
        print(f"       저장: {vis_output_single.name}")
        
        # 처리
        vis_output = vis_dir / "02_all_matches.jpg"
        _visualize_matches(img0, img1, mkpts0, mkpts1, all_mask, vis_output)
        print(f"       저장: {vis_output.name}")
        
        # 처리
        vis_output_grid = vis_dir / "03_grid_view.jpg"
        _visualize_matches_grid(img0, img1, pts0_in, pts1_in, inlier_only_mask, vis_output_grid, max_matches=20)
        print(f"       저장: {vis_output_grid.name}")
        
        # 처리
        vis_output_seq = vis_dir / "04_sequential_combined.jpg"
        _visualize_matches_sequential_combined(img0, img1, pts0_in, pts1_in, inlier_only_mask, vis_output_seq)
        print(f"       저장: {vis_output_seq.name}")
        
        # 처리
        _visualize_matches_sequential(img0, img1, pts0_in, pts1_in, inlier_only_mask, vis_dir, idx)
        print(f"       저장: 순 시각화 이미지들 (step by step)")


    return {
        **base,
        "status":        "ok",
        "matcher_used":  matcher_used,
        "raw_matches":   raw_n,
        "inlier_matches": inlier_n,
        "inlier_ratio":  inlier_n / raw_n,
        "control_points": control_points,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
          description="이미지 쌍의 특징점 매칭: 앵커 위치 LoFTR + MAGSAC++",
    )
    p.add_argument("--anchors-json", required=True,
                   help="05_anchor_patches.py의 앵커 JSON")
    p.add_argument("--confidence", type=float, default=0.2,
                   help="LoFTR 신뢰도 임계값 (기본값: 0.2)")
    p.add_argument("--magsac-thr", type=float, default=3.0,
                   help="MAGSAC++ 재투영 임계값 (픽셀, 기본값: 3.0)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/05.loftr/",
                   help="출력 디렉토리 (기본값: ./path-to-gpx/output/05.loftr/)")
    p.add_argument("--save-visualizations", action="store_true", default=True,
                   help="매칭 시각화 이미지 저장 (기본값: True)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    anchors_path = Path(args.anchors_json)
    if not anchors_path.exists():
        print(f"[ERROR] 앵커 JSON을 찾을 수 없음: {anchors_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data    = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = data["anchors"]
    stem    = Path(data["image"]).stem

    print(f"[INFO] {anchors_path.name}에서 {len(anchors)}개의 앵커 쌍 처리 중")
    if _MAGSAC is not None:
        print("[INFO] MAGSAC++ 알고리즘 (cv2.USAC_MAGSAC)")
    else:
        print("[INFO] cv2.USAC_MAGSAC을 찾을 수 없음 - RANSAC으로 대체")

    # 처리
    visualization_dir = None
    if args.save_visualizations:
        visualization_dir = output_dir / f"{stem}_match_visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 매칭 시각화 완료: {visualization_dir}")

    results: list[dict] = []
    total_cps = 0

    for i, anchor in enumerate(anchors):
        print(f"\n[{i+1}/{len(anchors)}] 앵커 idx={anchor['idx']}  '{anchor.get('text', '')}'")
        res = _process_anchor(anchor, args.confidence, args.magsac_thr, visualization_dir)
        results.append(res)
        n_cps = len(res.get("control_points", []))
        total_cps += n_cps
        print(f"    → 상태={res['status']}  제어점={n_cps}")

    out_json = output_dir / f"{stem}_loftr_matches.json"
    out_json.write_text(json.dumps({
        "anchors_json":          str(anchors_path),
        "image":                 data["image"],
        "image_width":           data["image_width"],
        "image_height":          data["image_height"],
        "confidence_threshold":  args.confidence,
        "magsac_threshold":      args.magsac_thr,
        "total_control_points":  total_cps,
        "anchor_results":        results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[완료] {len(anchors)}개의 앵커에서 {total_cps}개의 제어점 → {out_json}")
    return 0 if total_cps > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

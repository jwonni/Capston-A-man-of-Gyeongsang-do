'''
마라톤 경로 마스킹 이미지를 스켈레톤화하고 픽셀 좌표를 GPS 좌표로 변환하는 스크립트

- 입력: 마라톤 경로 마스킹 이미지, TPS/아핀 모델 JSON (06_tps_align.py 출력)
- 출력: 스켈레톤 이미지, GPS 웨이포인트 JSON, GPX 파일, 시각화 이미지

여기까지 흐름:
  1. OCR          → 텍스트 후보 추출
  2. Geocoding    → 텍스트 후보를 실제 좌표로 변환
  3. Map Tiles    → 좌표 기준 Naver 지도 이미지 다운로드
  4. Anchor Patches → 앵커 패치 추출 + Naver Maps 참조 타일 다운로드
  5. LoFTR Match  → 앵커 패치 쌍의 특징점 매칭 → 제어점 생성
  6. TPS Align    → 제어점으로 TPS/아핀 픽셀→GPS 모델 구축
  7. 이 스크립트  → 마스킹 이미지 스켈레톤화 → 경로 정렬 → 픽셀→GPS 변환 → GPX 생성
'''
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 스켈레톤화
# ---------------------------------------------------------------------------

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    '''
    이진 마스크를 1픽셀 중심선(스켈레톤)으로 변환한다.

    우선 순위:
      1. scikit-image Zhang-Suen 알고리즘 (설치된 경우)
      2. 반복 형태학적 침식 폴백 (scikit-image 미설치 시)

    매개변수
    --------
    mask : uint8 이진 마스크 (0 / 255 또는 0 / 1)

    반환
    ----
    uint8 스켈레톤 이미지 (0 / 255)
    '''
    try:
        from skimage.morphology import skeletonize as _sk_fn
        # skimage는 bool 배열을 입력으로 받음
        skel_bool = _sk_fn(mask > 0)
        return skel_bool.astype(np.uint8) * 255
    except ImportError:
        pass

    # scikit-image 미설치 시 반복 침식으로 대체
    # 원리: 전경 픽셀을 반복 침식하되, 매 단계에서 제거된 픽셀을 스켈레톤에 누적
    skel = np.zeros_like(mask, dtype=np.uint8)
    temp = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # 침식 → 팽창 → 원본과의 차분 = 이번 단계에서 제거된 테두리 픽셀
        eroded  = cv2.erode(temp, kernel)
        dilated = cv2.dilate(eroded, kernel)
        diff    = cv2.subtract(temp, dilated)
        skel    = cv2.bitwise_or(skel, diff)
        temp    = eroded
        # 남은 전경 픽셀이 없으면 종료
        if cv2.countNonZero(temp) == 0:
            break

    return skel


# ---------------------------------------------------------------------------
# 형태학적 전처리
# ---------------------------------------------------------------------------

def preprocess_mask(mask: np.ndarray, close_px: int) -> np.ndarray:
    '''
    마스킹 이미지를 스켈레톤화 전에 정제한다.

    처리 순서:
      1. 이진화 (임계값 127)
      2. 모폴로지 닫힘(Closing) — 경로의 끊어진 부분을 이어붙임
      3. 모폴로지 열림(Opening) — 작은 노이즈 제거
      4. 최대 연결 컴포넌트만 유지 — 화살표·레이블 등 무관한 객체 제거

    매개변수
    --------
    mask    : 입력 마스킹 이미지 (그레이스케일 또는 BGR)
    close_px: 닫힘 커널 반지름 (픽셀). 클수록 더 넓은 간격을 이어붙임
    '''
    # 다채널이면 그레이스케일로 변환
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    # 이진화
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 닫힘 커널 생성 (타원형이 직사각형보다 자연스러운 결과를 줌)
    diameter = 2 * close_px + 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k_open)

    # 가장 큰 연결 컴포넌트만 유지 (경로 외의 잡음 제거)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:
        # 배경만 있는 경우 그대로 반환
        return opened

    # 레이블 0은 배경이므로 1부터 비교
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == biggest).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# 경로 정렬 (스켈레톤 픽셀 순서화)
# ---------------------------------------------------------------------------

def _build_adjacency(pts_set: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    '''
    8방향 인접 딕셔너리를 구성한다. O(8N) 시간, O(1) 멤버십 탐색.

    매개변수
    --------
    pts_set : (x, y) 튜플의 집합 (스켈레톤 픽셀 좌표)

    반환
    ----
    {점: [인접 점 리스트]} 형태의 딕셔너리
    '''
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (x, y) in pts_set:
        # 8방향 이웃 중 pts_set에 포함된 점만 수집
        adj[(x, y)] = [
            (x + dx, y + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts_set
        ]
    return adj


def order_skeleton(skel: np.ndarray) -> list[tuple[int, int]]:
    '''
    스켈레톤 픽셀을 연속 경로 순서로 정렬한다.

    알고리즘
    --------
    1. 8방향 인접 딕셔너리 구성
    2. 차수 1인 끝점 탐색 → 끝점에서 탐색 시작
    3. 탐욕 탐색: 미방문 이웃 중 자신이 가장 적은 이웃을 가진 점 선택
       (분기점 대신 직선 경로를 우선함)
    4. 막힌 구간(dead-end): KDTree 최근접 이웃으로 도약하여 간격 극복

    매개변수
    --------
    skel : uint8 스켈레톤 이미지 (0 / 255)

    반환
    ----
    (x, y) 튜플의 순서 정렬된 리스트
    '''
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return []

    # 전체 스켈레톤 픽셀 집합
    all_pts  = set(zip(xs.tolist(), ys.tolist()))
    adj      = _build_adjacency(all_pts)
    unvisited: set[tuple[int, int]] = set(all_pts)

    # 차수 1 (이웃이 하나뿐인) 끝점에서 시작 — 없으면 임의 시작
    endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    start     = endpoints[0] if endpoints else next(iter(all_pts))

    ordered: list[tuple[int, int]] = [start]
    unvisited.discard(start)

    # KDTree 가용 여부 확인 (scipy 미설치 시 순차 탐색으로 대체)
    try:
        from scipy.spatial import KDTree
        _use_kdtree = True
    except ImportError:
        _use_kdtree = False

    while unvisited:
        cur = ordered[-1]
        # 현재 위치의 미방문 이웃 목록
        unvis_nbrs = [n for n in adj[cur] if n in unvisited]

        if unvis_nbrs:
            # 이웃의 이웃 수가 가장 적은 점을 선택 (직선 방향 유지 선호)
            nxt = min(
                unvis_nbrs,
                key=lambda n: sum(1 for m in adj[n] if m in unvisited),
            )
            ordered.append(nxt)
            unvisited.discard(nxt)
        else:
            # 막힌 구간: 미방문 픽셀 중 가장 가까운 점으로 도약
            uv_list = list(unvisited)
            uv_arr  = np.array(uv_list, dtype=np.int32)  # shape (M, 2) [x, y]

            if _use_kdtree:
                _, idx = KDTree(uv_arr).query(np.array(cur, dtype=np.int32))
                nearest = tuple(uv_arr[idx].tolist())
            else:
                # scipy 없을 때 O(M) 선형 탐색
                cx, cy = cur
                nearest = min(uv_list, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

            ordered.append(nearest)
            unvisited.discard(nearest)

    return ordered


def subsample_path(path: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    '''
    정렬된 경로를 균등 간격으로 서브샘플링한다.

    매개변수
    --------
    path   : 정렬된 (x, y) 픽셀 리스트
    target : 목표 웨이포인트 수

    반환
    ----
    서브샘플링된 (x, y) 리스트 (len ≤ target)
    '''
    n = len(path)
    if n <= target:
        return path
    # 인덱스를 균등 간격으로 선택
    indices = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
    return [path[i] for i in indices]


# ---------------------------------------------------------------------------
# TPS/아핀 모델 복원 및 픽셀→GPS 변환
# ---------------------------------------------------------------------------

def load_tps_model(tps_json_path: Path):
    '''
    06_tps_align.py가 저장한 TPS 모델 JSON을 읽어 변환 함수를 재구성한다.

    JSON 구조:
      {
        "model_type": "tps" | "affine",
        "smoothing":  float,
        "control_points": [{"img_x": ..., "img_y": ..., "lat": ..., "lon": ...}, ...]
      }

    반환
    ----
    (predict_fn, model_type)
      predict_fn : (M, 2) float64 배열 [img_x, img_y]를 받아 (M, 2) [lat, lon]을 반환
      model_type : "tps" 또는 "affine"
    '''
    data = json.loads(tps_json_path.read_text(encoding="utf-8"))
    cps  = data.get("control_points", [])

    if len(cps) < 2:
        raise RuntimeError(
            f"TPS 모델에 제어점이 부족합니다 ({len(cps)}개). "
            "06_tps_align.py를 먼저 실행하여 충분한 제어점을 생성하세요."
        )

    # 제어점을 numpy 배열로 변환
    src = np.array([[c["img_x"], c["img_y"]] for c in cps], dtype=np.float64)
    dst = np.array([[c["lat"],   c["lon"]]   for c in cps], dtype=np.float64)

    model_type = data.get("model_type", "affine")
    smoothing  = float(data.get("smoothing", 1e-4))

    if model_type == "tps" and len(cps) >= 4:
        try:
            from scipy.interpolate import RBFInterpolator
            # 저장된 것과 동일한 설정으로 TPS 모델 재구성
            rbf = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)
            print(f"  [INFO] TPS 모델 복원 완료 (제어점 {len(cps)}개, smoothing={smoothing})")
            return rbf, "tps"
        except Exception as exc:
            print(f"  [WARN] TPS 복원 실패 ({exc}), 아핀으로 대체합니다.")

    # 아핀 최소제곱 대체 모델
    # 증강 행렬: [x, y, 1] @ A ≈ [lat, lon]
    A_src = np.column_stack([src, np.ones(len(src))])
    A, _, _, _ = np.linalg.lstsq(A_src, dst, rcond=None)

    def affine_predict(query: np.ndarray) -> np.ndarray:
        '''아핀 변환: (M, 2) [img_x, img_y] → (M, 2) [lat, lon]'''
        q_aug = np.column_stack([query, np.ones(len(query))])
        return q_aug @ A

    print(f"  [INFO] 아핀 모델 복원 완료 (제어점 {len(cps)}개)")
    return affine_predict, "affine"


def pixels_to_gps(
    waypoints: list[tuple[int, int]],
    predict_fn,
) -> list[dict[str, Any]]:
    '''
    픽셀 좌표 리스트를 GPS 좌표(위도, 경도)로 일괄 변환한다.

    매개변수
    --------
    waypoints  : [(px, py), ...] 픽셀 좌표 리스트
    predict_fn : TPS/아핀 모델의 예측 함수

    반환
    ----
    [{"px": int, "py": int, "lat": float, "lon": float}, ...] 리스트
    '''
    if not waypoints:
        return []

    # (M, 2) 배열로 변환하여 일괄 예측 (반복 호출보다 훨씬 빠름)
    px_arr = np.array([[float(x), float(y)] for x, y in waypoints], dtype=np.float64)
    gps_arr = predict_fn(px_arr)  # shape (M, 2) — [lat, lon]

    result: list[dict[str, Any]] = []
    for i, (px, py) in enumerate(waypoints):
        lat = float(gps_arr[i, 0])
        lon = float(gps_arr[i, 1])
        result.append({"px": int(px), "py": int(py), "lat": lat, "lon": lon})

    return result


# ---------------------------------------------------------------------------
# GPX 생성
# ---------------------------------------------------------------------------

def build_gpx(waypoints_gps: list[dict[str, Any]], name: str) -> str:
    '''
    GPS 웨이포인트 리스트로부터 GPX 1.1 XML 문자열을 생성한다.

    매개변수
    --------
    waypoints_gps : [{"lat": float, "lon": float}, ...] 리스트
    name          : GPX 트랙 이름 (보통 이미지 stem)

    반환
    ----
    GPX XML 문자열
    '''
    now_iso = datetime.now(timezone.utc).isoformat()

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1"',
        '     creator="marathon-path-to-gpx"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 '
        'http://www.topografix.com/GPX/1/1/gpx.xsd">',
        f'  <metadata><name>{name}</name><time>{now_iso}</time></metadata>',
        '  <trk>',
        f'    <name>{name}</name>',
        '    <trkseg>',
    ]

    for wp in waypoints_gps:
        lat = wp["lat"]
        lon = wp["lon"]
        # 유효 범위를 벗어난 좌표는 GPX에 포함하지 않음
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        lines.append(f'      <trkpt lat="{lat:.8f}" lon="{lon:.8f}"><ele>0</ele></trkpt>')

    lines += [
        '    </trkseg>',
        '  </trk>',
        '</gpx>',
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def visualize_gps_on_mask(
    mask: np.ndarray,
    waypoints_gps: list[dict[str, Any]],
) -> np.ndarray:
    '''
    마스킹 이미지 위에 GPS 변환이 완료된 픽셀 좌표를 빨간색으로 표시한다.

    시각화 내용:
      - 마스킹 이미지를 BGR로 변환하여 배경으로 사용
      - 웨이포인트를 연결하는 빨간색 폴리라인
      - 각 웨이포인트 위치에 빨간색 원
      - 시작점(초록), 끝점(파랑) 표시
      - 좌상단에 통계 텍스트 오버레이

    매개변수
    --------
    mask          : 원본 마스킹 이미지 (그레이스케일 또는 BGR)
    waypoints_gps : [{"px": int, "py": int, "lat": float, "lon": float}, ...] 리스트

    반환
    ----
    시각화된 BGR 이미지
    '''
    # 그레이스케일 마스크를 BGR로 변환하여 컬러 오버레이 가능하게 함
    if mask.ndim == 2:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = mask.copy()

    if not waypoints_gps:
        return vis

    # 웨이포인트 픽셀 좌표 추출
    pts = np.array([[wp["px"], wp["py"]] for wp in waypoints_gps], dtype=np.int32)
    n   = len(pts)

    # 빨간색 폴리라인으로 경로 연결 (GPS 변환된 픽셀들의 순서를 표현)
    cv2.polylines(
        vis,
        [pts.reshape(-1, 1, 2)],
        isClosed=False,
        color=(0, 0, 255),       # 빨간색 (BGR)
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # 각 웨이포인트에 빨간 원 표시 (점이 너무 많으면 간격을 두어 표시)
    dot_step = max(1, n // 200)   # 최대 200개 점만 표시하여 가시성 확보
    for i in range(0, n, dot_step):
        cv2.circle(vis, tuple(pts[i]), 3, (0, 0, 220), -1)

    # 시작점 (초록색 큰 원)
    cv2.circle(vis, tuple(pts[0]),  9, (0, 220, 0),  -1)
    cv2.circle(vis, tuple(pts[0]),  9, (0, 180, 0),   2)

    # 끝점 (파란색 큰 원)
    cv2.circle(vis, tuple(pts[-1]), 9, (220, 50,  0), -1)
    cv2.circle(vis, tuple(pts[-1]), 9, (180, 30,  0),  2)

    # 통계 텍스트 오버레이
    lat_vals = [wp["lat"] for wp in waypoints_gps]
    lon_vals = [wp["lon"] for wp in waypoints_gps]
    info_lines = [
        f"Waypoints: {n}",
        f"Lat: {min(lat_vals):.5f} ~ {max(lat_vals):.5f}",
        f"Lon: {min(lon_vals):.5f} ~ {max(lon_vals):.5f}",
        "Start: Green  End: Blue  Path: Red",
    ]
    # 텍스트 배경 박스
    box_h = len(info_lines) * 22 + 12
    overlay = vis.copy()
    cv2.rectangle(overlay, (8, 8), (420, 8 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    for i, line in enumerate(info_lines):
        cv2.putText(
            vis, line,
            (14, 30 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )

    return vis


# ---------------------------------------------------------------------------
# OSRM Road Snapping (도로 네트워크 스냅)
# ---------------------------------------------------------------------------

def _osrm_match_batch(
    batch: list[dict],
    osrm_url: str,
    profile: str,
    timeout: int = 30,
) -> list[dict] | None:
    '''
    OSRM Map Matching API로 웨이포인트 배치를 도로에 스냅한다.

    OSRM match 서비스는 여러 GPS 포인트를 한 번의 요청으로 도로 네트워크에
    맞춰 정렬해준다. 경로의 연속성을 고려하므로 nearest보다 자연스러운 결과를 냄.

    반환: 스냅된 웨이포인트 리스트, API 오류 시 None 반환 (nearest 폴백용)

    주의: OSRM의 location 필드는 [경도, 위도] 순서 (GPS와 반대)
    '''
    coords = ";".join(f"{wp['lon']:.8f},{wp['lat']:.8f}" for wp in batch)
    url = (
        f"{osrm_url.rstrip('/')}/match/v1/{profile}/{coords}"
        f"?gaps=split&geometries=geojson&overview=false&annotations=false"
    )

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "marathon-path-to-gpx/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    if data.get("code") != "Ok":
        return None

    tracepoints = data.get("tracepoints")
    # tracepoints 길이가 배치 크기와 다르면 신뢰 불가
    if not tracepoints or len(tracepoints) != len(batch):
        return None

    result = []
    for wp, tp in zip(batch, tracepoints):
        if tp is None:
            # 이 포인트는 도로를 찾지 못함 → 원본 좌표 유지
            result.append({
                **wp,
                "snapped":     False,
                "snap_dist_m": None,
                "road_name":   "",
            })
        else:
            # OSRM location은 [경도, 위도] 순서
            snapped_lon, snapped_lat = tp["location"]
            result.append({
                **wp,
                "lat":         snapped_lat,
                "lon":         snapped_lon,
                "orig_lat":    wp["lat"],
                "orig_lon":    wp["lon"],
                "snapped":     True,
                "snap_dist_m": round(float(tp.get("distance", 0.0)), 2),
                "road_name":   tp.get("name", ""),
            })
    return result


def _osrm_nearest_single(
    wp: dict,
    osrm_url: str,
    profile: str,
    timeout: int = 10,
) -> dict:
    '''
    OSRM Nearest API로 단일 웨이포인트를 가장 가까운 도로 위의 점에 스냅한다.

    match 서비스 실패 시 폴백으로 사용된다.
    모든 네트워크/응답 오류에서 원본 좌표를 유지한 딕셔너리를 반환한다.
    '''
    url = (
        f"{osrm_url.rstrip('/')}/nearest/v1/{profile}"
        f"/{wp['lon']:.8f},{wp['lat']:.8f}?number=1"
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "marathon-path-to-gpx/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("code") == "Ok" and data.get("waypoints"):
            w = data["waypoints"][0]
            snapped_lon, snapped_lat = w["location"]  # [경도, 위도] 순서
            return {
                **wp,
                "lat":         snapped_lat,
                "lon":         snapped_lon,
                "orig_lat":    wp["lat"],
                "orig_lon":    wp["lon"],
                "snapped":     True,
                "snap_dist_m": round(float(w.get("distance", 0.0)), 2),
                "road_name":   w.get("name", ""),
            }
    except Exception:
        pass

    # 스냅 실패 시 원본 유지
    return {**wp, "snapped": False, "snap_dist_m": None, "road_name": ""}


def snap_to_roads_osrm(
    waypoints_gps: list[dict],
    osrm_url: str,
    profile: str = "foot",
    batch_size: int = 100,
    batch_delay_s: float = 0.3,
) -> list[dict]:
    '''
    OSRM Map Matching / Nearest 서비스로 GPS 웨이포인트를 도로 네트워크에 스냅한다.

    TPS/아핀 모델에서 변환된 GPS 좌표는 도로와 약간 어긋날 수 있다.
    이 함수는 각 포인트를 실제 도로 위로 이동시켜 GPX 경로를 정밀 보정한다.

    전략
    ----
    1. batch_size 단위로 분할하여 OSRM match 서비스 요청
       - 경로 연속성을 고려한 일괄 스냅 (nearest보다 자연스러운 결과)
       - 한 번의 HTTP 요청으로 여러 포인트를 처리 → 공개 서버 부하 최소화
    2. match 실패 시 해당 배치의 각 포인트를 nearest 서비스로 개별 스냅
       - 포인트 간 연속성은 없지만 단독으로는 더 안정적
    3. 모든 오류에서 원본 좌표 유지 (실패 허용 설계)

    매개변수
    --------
    waypoints_gps : [{"px", "py", "lat", "lon"}, ...] 웨이포인트 리스트
    osrm_url      : OSRM 서버 기본 URL (기본: "http://router.project-osrm.org")
    profile       : 이동 수단 프로파일 ("foot", "car", "bike")
    batch_size    : 배치당 최대 포인트 수 (공개 OSRM 서버 권장값: 100)
    batch_delay_s : 배치 간 딜레이 초 (공개 서버 Rate-limit 방지)

    반환
    ----
    스냅된 GPS 좌표로 업데이트된 웨이포인트 리스트.
    스냅된 포인트는 "orig_lat"/"orig_lon"으로 원본 좌표를 보존한다.
    '''
    n = len(waypoints_gps)
    if n == 0:
        return []

    result: list[dict] = []
    total_batches = (n + batch_size - 1) // batch_size

    for b_idx, batch_start in enumerate(range(0, n, batch_size)):
        batch = waypoints_gps[batch_start:batch_start + batch_size]
        print(f"  배치 {b_idx + 1}/{total_batches}  ({len(batch)}개 포인트) ...", end=" ", flush=True)

        # 1단계: match 서비스 시도 (배치 전체를 한 번에 스냅)
        snapped = _osrm_match_batch(batch, osrm_url, profile)

        if snapped is None:
            # 2단계: match 실패 → 포인트별 nearest 폴백
            print("match 실패, nearest 폴백 중 ...")
            snapped = []
            for wp in batch:
                snapped.append(_osrm_nearest_single(wp, osrm_url, profile))
                time.sleep(0.05)  # nearest는 포인트별 개별 요청이므로 짧은 딜레이
        else:
            n_ok = sum(1 for wp in snapped if wp.get("snapped"))
            print(f"스냅 {n_ok}/{len(batch)}")

        result.extend(snapped)

        # 마지막 배치가 아닐 때만 딜레이 (공개 서버 Rate-limit 방지)
        if b_idx < total_batches - 1:
            time.sleep(batch_delay_s)

    # 전체 통계 출력
    n_snapped = sum(1 for wp in result if wp.get("snapped", False))
    n_failed  = n - n_snapped
    if n_snapped > 0:
        dists = [wp["snap_dist_m"] for wp in result if wp.get("snap_dist_m") is not None]
        avg_d = sum(dists) / len(dists) if dists else 0.0
        print(
            f"  [결과] 스냅 성공={n_snapped}/{n}  실패={n_failed}  "
            f"평균 스냅 거리={avg_d:.1f}m"
        )
    else:
        print("  [결과] 모든 포인트 스냅 실패 — 원본 좌표를 유지합니다.")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="마라톤 경로 마스킹 이미지 스켈레톤화 + 픽셀→GPS 변환 + GPX 생성"
    )
    parser.add_argument(
        "--mask-image",
        required=True,
        help="마라톤 경로 마스킹 이미지 경로 (이진 또는 그레이스케일)",
    )
    parser.add_argument(
        "--tps-json",
        required=True,
        help="06_tps_align.py 출력 TPS/아핀 모델 JSON 경로",
    )
    parser.add_argument(
        "--close-px",
        type=int,
        default=5,
        help="닫힘 커널 반지름(px) — 마스크의 끊어진 부분을 이어붙임 (기본값: 5)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=500,
        help="경로 서브샘플링 목표 웨이포인트 수 (기본값: 500)",
    )
    parser.add_argument(
        "--output-dir",
        default="./path-to-gpx/output/07.skeleton_gps/",
        help="출력 디렉토리 (기본값: ./path-to-gpx/output/07.skeleton_gps/)",
    )
    # ── OSRM Road Snapping ────────────────────────────────────────────────
    parser.add_argument(
        "--no-road-snap",
        action="store_true",
        default=True, # 건너뛰기 너무 오래걸림
        help="OSRM 도로 스냅을 건너뜀 (기본값: 스냅 실행)",
    )
    parser.add_argument(
        "--osrm-url",
        default="http://router.project-osrm.org",
        help=(
            "OSRM 서버 기본 URL "
            "(기본값: http://router.project-osrm.org — 공개 데모 서버. "
            "로컬 서버 사용 시 http://localhost:5000 등으로 변경)"
        ),
    )
    parser.add_argument(
        "--osrm-profile",
        default="foot",
        choices=["foot", "car", "bike"],
        help="OSRM 이동 수단 프로파일 (기본값: foot — 마라톤에 적합)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args   = parser.parse_args()

    # ── 입력 파일 검증 ─────────────────────────────────────────────────────
    mask_path = Path(args.mask_image)
    tps_path  = Path(args.tps_json)

    for path, name in [(mask_path, "--mask-image"), (tps_path, "--tps-json")]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{name} 파일을 찾을 수 없음: {path}")

    close_px  = int(args.close_px)
    subsample = int(args.subsample)
    if subsample < 2:
        raise ValueError("--subsample은 2 이상이어야 합니다.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = mask_path.stem   # 출력 파일명 접두사

    # ── 마스킹 이미지 로드 ─────────────────────────────────────────────────
    print(f"[INFO] 마스킹 이미지 로드: {mask_path}")
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {mask_path}")

    h_img, w_img = raw_mask.shape[:2]
    print(f"[INFO] 이미지 크기: {w_img}×{h_img}")

    # ── 형태학적 전처리 ────────────────────────────────────────────────────
    print(f"[INFO] 마스크 전처리 (닫힘 커널={close_px}px) …")
    clean = preprocess_mask(raw_mask, close_px)
    mask_px = int(np.count_nonzero(clean))
    print(f"  전처리 후 전경 픽셀 수: {mask_px}")

    if mask_px == 0:
        raise RuntimeError(
            "전처리 후 전경 픽셀이 없습니다. "
            "마스킹 이미지가 올바른지, --close-px 값을 조정하세요."
        )

    # 전처리된 마스크 저장
    clean_path = output_dir / f"{stem}_clean_mask.png"
    cv2.imwrite(str(clean_path), clean)
    print(f"  전처리 마스크 → {clean_path.name}")

    # ── 스켈레톤화 ─────────────────────────────────────────────────────────
    print("[INFO] 스켈레톤화 (Zhang-Suen / 반복 침식) …")
    skel = skeletonize_mask(clean)
    skel_px = int(np.count_nonzero(skel))
    print(f"  스켈레톤 픽셀 수: {skel_px}")

    if skel_px < 10:
        raise RuntimeError(
            f"스켈레톤 픽셀이 너무 적습니다 ({skel_px}개). "
            "마스킹 이미지가 올바른지 확인하거나 --close-px를 늘려보세요."
        )

    # 스켈레톤 이미지 저장
    skel_path = output_dir / f"{stem}_skeleton.png"
    cv2.imwrite(str(skel_path), skel)
    print(f"  스켈레톤 → {skel_path.name}")

    # ── 경로 순서 정렬 ─────────────────────────────────────────────────────
    print("[INFO] 스켈레톤 픽셀 경로 순서 정렬 …")
    ordered = order_skeleton(skel)
    print(f"  정렬된 픽셀 수: {len(ordered)}")

    # 서브샘플링: 너무 많은 웨이포인트는 GPX를 비대하게 만듦
    waypoints_px = subsample_path(ordered, subsample)
    print(f"  서브샘플링 후 웨이포인트 수: {len(waypoints_px)}")

    # ── TPS/아핀 모델 로드 및 GPS 변환 ────────────────────────────────────
    print(f"[INFO] TPS 모델 로드: {tps_path}")
    predict_fn, model_type = load_tps_model(tps_path)
    print(f"[INFO] 모델 유형: {model_type}")

    print("[INFO] 픽셀 좌표 → GPS 변환 중 …")
    waypoints_gps = pixels_to_gps(waypoints_px, predict_fn)

    # 변환 결과 범위 출력 (품질 확인용)
    lats = [w["lat"] for w in waypoints_gps]
    lons = [w["lon"] for w in waypoints_gps]
    print(f"  위도 범위: {min(lats):.6f} ~ {max(lats):.6f}")
    print(f"  경도 범위: {min(lons):.6f} ~ {max(lons):.6f}")

    # ── OSRM Road Snapping ─────────────────────────────────────────────────
    # TPS 모델이 예측한 GPS 좌표를 실제 도로 위로 정밀 보정한다.
    # 스냅 전 원본 GPX를 먼저 저장해두어 비교가 가능하도록 한다.
    n_snapped = 0
    if not args.no_road_snap:
        print(
            f"[INFO] OSRM Road Snapping  "
            f"(서버: {args.osrm_url}  프로파일: {args.osrm_profile}) …"
        )

        # 스냅 전 원본 GPX 저장 (비교 및 롤백용)
        raw_gpx_path = output_dir / f"{stem}_route_raw.gpx"
        raw_gpx_path.write_text(
            build_gpx(waypoints_gps, name=f"{stem}_raw"), encoding="utf-8"
        )
        print(f"  원본 GPX (스냅 전) → {raw_gpx_path.name}")

        # 도로 스냅 실행
        waypoints_gps = snap_to_roads_osrm(
            waypoints_gps,
            osrm_url=args.osrm_url,
            profile=args.osrm_profile,
        )
        n_snapped = sum(1 for wp in waypoints_gps if wp.get("snapped", False))

        # 스냅 후 좌표 범위 재계산
        lats = [w["lat"] for w in waypoints_gps]
        lons = [w["lon"] for w in waypoints_gps]
        print(f"  스냅 후 위도 범위: {min(lats):.6f} ~ {max(lats):.6f}")
        print(f"  스냅 후 경도 범위: {min(lons):.6f} ~ {max(lons):.6f}")
    else:
        print("[INFO] OSRM Road Snapping 건너뜀 (--no-road-snap 옵션)")

    # ── 웨이포인트 JSON 저장 ───────────────────────────────────────────────
    tps_data      = json.loads(tps_path.read_text(encoding="utf-8"))
    n_cps         = tps_data.get("n_total_cps", len(tps_data.get("control_points", [])))
    residual_mean = tps_data.get("tps_self_residual_mean_deg", None)

    json_out = output_dir / f"{stem}_waypoints.json"
    json_out.write_text(json.dumps({
        "created_at_utc":        datetime.now(timezone.utc).isoformat(),
        "mask_image":            str(mask_path),
        "image_width":           w_img,
        "image_height":          h_img,
        "tps_json":              str(tps_path),
        "model_type":            model_type,
        "n_control_points":      n_cps,
        "tps_residual_mean_deg": residual_mean,
        "skeleton_px":           skel_px,
        "ordered_px":            len(ordered),
        "waypoint_count":        len(waypoints_gps),
        "lat_range":             [min(lats), max(lats)],
        "lon_range":             [min(lons), max(lons)],
        # OSRM Road Snapping 메타데이터
        "road_snap_applied":     not args.no_road_snap,
        "osrm_url":              args.osrm_url if not args.no_road_snap else None,
        "osrm_profile":          args.osrm_profile if not args.no_road_snap else None,
        "n_snapped":             n_snapped,
        "waypoints":             waypoints_gps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 웨이포인트 JSON → {json_out.name}")

    # ── GPX 파일 생성 ──────────────────────────────────────────────────────
    print("[INFO] GPX 파일 생성 …")
    gpx_str  = build_gpx(waypoints_gps, name=stem)
    gpx_path = output_dir / f"{stem}_route.gpx"
    gpx_path.write_text(gpx_str, encoding="utf-8")
    valid_trkpts = gpx_str.count("<trkpt")
    print(f"[DONE] GPX → {gpx_path.name}  (트랙포인트: {valid_trkpts}개)")

    # ── 시각화 ────────────────────────────────────────────────────────────
    print("[INFO] GPS 변환 픽셀 위치 시각화 …")
    vis = visualize_gps_on_mask(raw_mask, waypoints_gps)
    vis_path = output_dir / f"{stem}_gps_visualized.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"[DONE] 시각화 → {vis_path.name}")

    snap_note = (
        f"  도로 스냅: {n_snapped}/{len(waypoints_gps)} 포인트"
        if not args.no_road_snap else "  도로 스냅: 건너뜀"
    )
    print(
        f"\n[완료] 웨이포인트 {len(waypoints_gps)}개 → "
        f"JSON: {json_out.name}, GPX: {gpx_path.name}, 시각화: {vis_path.name}\n"
        f"{snap_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
픽셀 경로를 GPX 파일로 변환하는 유틸리티.

입력:
- start: 시작점 좌표 [x, y]
- end: 끝점 좌표 [x, y]
- path: 시작점부터 끝점까지 정렬된 픽셀 좌표 목록 [[x, y], ...]
- pixel_to_geo: (px, py) → (lat, lng) 변환 함수 (None이면 픽셀을 그대로 사용)

출력:
- GPX 1.1 형식의 XML 문자열
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import atan2, cos, radians, sin, sqrt
from typing import Callable
from xml.sax.saxutils import escape

import numpy as np


def convert_pixel_path_to_gpx(
    start: list[int | float],
    end: list[int | float],
    path: list[list[int | float]],
    creator: str = "Marathon Route Extractor",
    pixel_to_geo: Callable[[float, float], tuple[float, float]] | None = None,
) -> str:
    """픽셀 경로를 GPX 1.1 문자열로 변환한다.

    pixel_to_geo(px, py) → (lat, lng) 가 주어지면 실제 지리좌표를 사용하고,
    없으면 픽셀 좌표를 위도/경도로 그대로 사용한다 (하위 호환).
    """
    if len(start) != 2 or len(end) != 2:
        raise ValueError("start/end must be [x, y]")
    if not path:
        raise ValueError("path is required")

    points = [start, *path, end]
    base_time = datetime.now(timezone.utc)
    trkpts = []
    for i, (x, y) in enumerate(points):
        if pixel_to_geo is not None:
            lat, lng = pixel_to_geo(float(x), float(y))
        else:
            lat, lng = float(y), float(x)
        t = (base_time + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        trkpts.append(
            f'      <trkpt lat="{lat:.8f}" lon="{lng:.8f}">\n'
            f'        <time>{t}</time>\n'
            f'      </trkpt>'
        )

    track_name   = escape("Marathon Route")
    creator_name = escape(creator)

    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<gpx version="1.1" creator="{creator_name}"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1',
        '     http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <metadata>",
        f"    <name>{track_name}</name>",
        f"    <time>{base_time.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>",
        "  </metadata>",
        "  <trk>",
        f"    <name>{track_name}</name>",
        "    <trkseg>",
        *trkpts,
        "    </trkseg>",
        "  </trk>",
        "</gpx>",
    ])


# ── 위경도 route dict 기반 GPX 유틸 ──────────────────────────────────────────

def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """두 위경도 점 사이 거리(m). p1/p2 = (lat, lng)"""
    R = 6371000
    lat1, lng1 = radians(p1[0]), radians(p1[1])
    lat2, lng2 = radians(p2[0]), radians(p2[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def has_jumps(
    route: list[dict],
    threshold_m: float = 50.0,
) -> tuple[bool, int, float]:
    """연속된 경로 점 사이에 threshold_m 초과 점프가 있는지 확인한다.

    Returns:
        (found, index, distance_m) — 점프가 없으면 (False, -1, 0)
    """
    if len(route) < 2:
        return False, -1, 0.0
    R = 6_371_000
    for i in range(len(route) - 1):
        lat1, lng1 = route[i]["lat"], route[i]["lng"]
        lat2, lng2 = route[i + 1]["lat"], route[i + 1]["lng"]
        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lng2 - lng1)
        a = (np.sin(d_lat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(d_lon / 2) ** 2)
        d = 2 * R * np.arcsin(np.sqrt(a))
        if d > threshold_m:
            return True, i, float(d)
    return False, -1, 0.0


def reorder_route_greedy(
    route: list[dict],
    start_pixel: tuple[int, int] | None = None,
    end_pixel:   tuple[int, int] | None = None,
) -> list[dict]:
    """KDTree 기반 greedy nearest-neighbor로 경로 포인트를 재정렬한다.

    Args:
        route:       [{"pixel_x", "pixel_y", "lat", "lng"}, ...] 목록
        start_pixel: 시작점 픽셀 좌표. None이면 (x+y) 최솟값 점을 사용.
        end_pixel:   끝점 픽셀 좌표. 이 점에 3px 이내 도달하면 탐색을 종료.

    Returns:
        재정렬된 동일 구조의 목록
    """
    from scipy.spatial import KDTree

    pts = np.array([[r["pixel_x"], r["pixel_y"]] for r in route], dtype=np.float64)
    n = len(pts)
    if n == 0:
        return []

    if start_pixel is not None:
        start = int(np.argmin(np.linalg.norm(pts - np.array(start_pixel), axis=1)))
    else:
        start = int(np.argmin(pts[:, 0] + pts[:, 1]))

    visited = np.zeros(n, dtype=bool)
    order:  list[int] = []
    cur     = start
    tree    = KDTree(pts)

    for _ in range(n):
        visited[cur] = True
        order.append(cur)

        if end_pixel is not None and np.linalg.norm(pts[cur] - np.array(end_pixel)) < 3:
            break

        k = min(20, n)
        _, idxs = tree.query(pts[cur], k=k)
        idxs = np.atleast_1d(idxs)
        nxt  = None
        for idx in idxs:
            if not visited[int(idx)]:
                nxt = int(idx)
                break
        if nxt is None:
            remaining = np.where(~visited)[0]
            if len(remaining) == 0:
                break
            dists = np.linalg.norm(pts[remaining] - pts[cur], axis=1)
            if dists.min() > 100:
                break
            nxt = int(remaining[np.argmin(dists)])
        cur = nxt

    return [route[i] for i in order]


def check_route_quality(route: list[dict]) -> dict:
    """경로 품질 지표를 계산한다.

    Args:
        route: [{"lat", "lng"}, ...] 목록

    Returns:
        {total_km, straight_km, curve_ratio, max_gap_m, mean_gap_m, jumps_over_100m}
    """
    if len(route) < 2:
        return {"total_km": 0.0, "straight_km": 0.0, "curve_ratio": 0.0,
                "max_gap_m": 0.0, "mean_gap_m": 0.0, "jumps_over_100m": 0}
    ll    = [(p["lat"], p["lng"]) for p in route]
    dists = np.array([haversine_m(ll[i], ll[i + 1]) for i in range(len(ll) - 1)])
    total_km    = float(dists.sum() / 1000)
    straight_km = haversine_m(ll[0], ll[-1]) / 1000
    curve_ratio = total_km / straight_km if straight_km > 0 else 0.0
    return {
        "total_km":       round(total_km, 3),
        "straight_km":    round(straight_km, 3),
        "curve_ratio":    round(curve_ratio, 2),
        "max_gap_m":      round(float(dists.max()), 2),
        "mean_gap_m":     round(float(dists.mean()), 2),
        "jumps_over_100m": int((dists > 100).sum()),
    }


def build_gpx(route: list[dict], track_name: str = "Marathon Route") -> str:
    """위경도 route dict 목록 → GPX 1.1 문자열.

    Args:
        route:      [{"lat", "lng"}, ...] (범위 외 좌표는 자동 제외)
        track_name: 트랙 이름

    Returns:
        GPX XML 문자열
    """
    base_time = datetime.utcnow()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="MarathonOCR"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1'
        ' http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <metadata>",
        f"    <name>{track_name}</name>",
        f"    <time>{base_time.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>",
        "  </metadata>",
        "  <trk>",
        f"    <name>{track_name}</name>",
        "    <trkseg>",
    ]
    for i, pt in enumerate(route):
        lat = float(pt["lat"])
        lng = float(pt.get("lng", pt.get("lon", 0.0)))
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        t = (base_time + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(f'      <trkpt lat="{lat:.8f}" lon="{lng:.8f}">')
        lines.append(f"        <time>{t}</time>")
        lines.append("      </trkpt>")
    lines += ["    </trkseg>", "  </trk>", "</gpx>"]
    return "\n".join(lines)


# ── white_line ↔ OCR 카테고리 매칭 → 방향 보정 / 왕복 path 생성 ────────────────

def fix_white_line_path(
    white_line: dict,
    category_pixels: dict,
) -> dict:
    """OCR 카테고리 정보로 white_line 경로 방향을 보정하고 왕복 여부를 판단한다.

    Args:
        white_line:       {"start": [x,y], "end": [x,y], "path": [[x,y], ...]}
        category_pixels:  {"start_finish": [...], "turning_point": [...]}
                          각 원소는 {"x": int, "y": int, "text": str, ...} 형태.

    Returns:
        {"start", "end", "path", "meta"} dict.
        path 는 최종 순서대로 정렬된 전체 픽셀 좌표 목록.
    """
    def _dist(a: tuple, b: tuple) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    wl_start: tuple = tuple(white_line["start"])
    wl_end:   tuple = tuple(white_line["end"])
    wl_path:  list  = [tuple(p) for p in white_line["path"]]

    start_finish_pts  = category_pixels.get("start_finish", [])
    turning_point_pts = category_pixels.get("turning_point", [])

    # ── 방향 결정 ────────────────────────────────────────────────────────────────
    # start_finish OCR 점 전체를 사용해 wl_start / wl_end 중 어느 쪽이
    # 출발/도착선에 더 가까운지 판단한다 (단일 [0] 의존 버그 수정).
    used_ocr_anchor = False
    if start_finish_pts:
        sf_coords = [(sf["x"], sf["y"]) for sf in start_finish_pts]
        d_sf_to_start = min(_dist(p, wl_start) for p in sf_coords)
        d_sf_to_end   = min(_dist(p, wl_end)   for p in sf_coords)

        if d_sf_to_end < d_sf_to_start:
            # 출발/도착선이 wl_end 쪽에 더 가까움 → 역순
            wl_path = wl_path[::-1]
            wl_start, wl_end = wl_end, wl_start

        used_ocr_anchor = True

    # ── 왕복 경로 생성 ───────────────────────────────────────────────────────────
    has_turning = len(turning_point_pts) > 0
    if has_turning:
        forward  = list(wl_path)
        backward = list(wl_path[-2::-1])   # 반환점(마지막 점) 제외 역순
        final_path = forward + backward
        final_end  = wl_path[0]            # 출발점으로 귀환
    else:
        final_path = list(wl_path)
        final_end  = wl_path[-1]

    return {
        "start": list(wl_path[0]),
        "end":   list(final_end),
        "path":  [list(p) for p in final_path],
        "meta": {
            "is_round_trip":       has_turning,
            "used_ocr_anchor":     used_ocr_anchor,
        },
    }


def fix_white_line_path_from_files(
    white_line_json: str,
    category_pixels_json: str,
    output_json: str | None = None,
) -> dict:
    """파일 경로를 받아 fix_white_line_path 를 실행하고 결과를 반환한다.

    Args:
        white_line_json:      white_line_coords.json 경로
        category_pixels_json: category_pixels.json 경로
        output_json:          결과를 저장할 경로 (None 이면 저장 안 함)

    Returns:
        fix_white_line_path 의 반환값과 동일한 dict.
    """
    with open(white_line_json, encoding="utf-8") as f:
        wl = json.load(f)
    with open(category_pixels_json, encoding="utf-8") as f:
        cat = json.load(f)

    result = fix_white_line_path(wl, cat)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result

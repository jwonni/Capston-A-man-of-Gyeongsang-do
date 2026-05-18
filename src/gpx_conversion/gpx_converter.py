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


def reorder_route_greedy(route: list[dict]) -> list[dict]:
    """KDTree 기반 greedy nearest-neighbor로 경로 포인트를 재정렬한다.

    Args:
        route: [{"pixel_x", "pixel_y", "lat", "lng"}, ...] 목록

    Returns:
        재정렬된 동일 구조의 목록
    """
    from scipy.spatial import KDTree

    pts = np.array([[r["pixel_x"], r["pixel_y"]] for r in route], dtype=np.float64)
    n = len(pts)
    if n == 0:
        return []

    start   = int(np.argmin(pts[:, 0] + pts[:, 1]))
    visited = np.zeros(n, dtype=bool)
    order:  list[int] = []
    cur     = start
    tree    = KDTree(pts)

    for _ in range(n):
        visited[cur] = True
        order.append(cur)
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
            nxt   = int(remaining[np.argmin(dists)])
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

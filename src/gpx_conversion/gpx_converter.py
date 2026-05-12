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
from typing import Callable
from xml.sax.saxutils import escape


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

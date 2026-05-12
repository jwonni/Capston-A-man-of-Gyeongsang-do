"""
픽셀 경로를 GPX 파일로 변환하는 유틸리티.

입력:
- start: 시작점 좌표 [x, y]
- end: 끝점 좌표 [x, y]
- path: 시작점부터 끝점까지 정렬된 픽셀 좌표 목록 [[x, y], ...]

출력:
- GPX 1.1 형식의 XML 문자열
"""

from __future__ import annotations

from xml.sax.saxutils import escape


def convert_pixel_path_to_gpx(
    start: list[int | float],
    end: list[int | float],
    path: list[list[int | float]],
    creator: str = "Marathon Route Extractor",
) -> str:
    """픽셀 경로를 GPX 1.1 문자열로 변환한다.

    입력:
    - start: [x, y] 시작점 좌표
    - end: [x, y] 끝점 좌표
    - path: [[x, y], ...] 순서가 보장된 경로 좌표
    - creator: GPX 메타데이터에 기록할 생성자 이름

    출력:
    - GPX XML 문자열
    """
    if len(start) != 2 or len(end) != 2:
        raise ValueError("start/end must be [x, y]")
    if not path:
        raise ValueError("path is required")

    points = [start, *path, end]

    trkpts = []
    for x, y in points:
        trkpts.append(f"      <trkpt lat=\"{float(y):.6f}\" lon=\"{float(x):.6f}\" />")

    track_name = escape("Marathon Route")
    creator_name = escape(creator)

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<gpx version="1.1" creator="{creator_name}">',
            "  <metadata>",
            f"    <name>{track_name}</name>",
            "  </metadata>",
            "  <trk>",
            f"    <name>{track_name}</name>",
            "    <trkseg>",
            *trkpts,
            "    </trkseg>",
            "  </trk>",
            "</gpx>",
        ]
    )

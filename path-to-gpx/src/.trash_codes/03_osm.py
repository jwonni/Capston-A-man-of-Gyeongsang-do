'''
OpenStreetMap 타일을 다운로드하여 지오코딩된 좌표에 대한 시각적 참조를 제공하는 스크립트
- 입력: 지오코딩 후보 JSON (예: *_geocode_candidates.json)
- 출력: 핵심 후보 기준 OSM 타일 이미지 1장 및 보고서 JSON
- 주요 기능:
	- 지오코딩 후보 중 핵심 키워드를 반영해 대표 좌표 1개 선택
	- 선택 좌표를 OSM 타일 좌표로 변환하여 타일 1장 다운로드
	- 다운로드된 타일과 메타데이터를 포함하는 보고서 생성
- 의존성: 없음 (표준 라이브러리만 사용)


여기까지 흐름은 다음과 같다:
1. ocr을 통해 텍스트 후보 추출
2. geocoding을 통해 텍스트 후보를 실제 좌표로 변환 
3. 이 스크립트에서 좌표를 OSM 타일로 변환하여 다운로드
'''
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import cv2
import numpy as np


@dataclass
class CoordinateCandidate:
	idx: int
	text: str
	matched_place: str
	lat: float
	lon: float
	display_name: str


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
	lat = max(min(lat, 85.05112878), -85.05112878)
	lon = ((lon + 180.0) % 360.0) - 180.0

	lat_rad = math.radians(lat)
	n = 2**zoom
	xtile = int((lon + 180.0) / 360.0 * n)
	ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
	return xtile, ytile


def latlon_to_tile_float(lat: float, lon: float, zoom: int) -> tuple[float, float]:
	lat = max(min(lat, 85.05112878), -85.05112878)
	lon = ((lon + 180.0) % 360.0) - 180.0

	lat_rad = math.radians(lat)
	n = 2**zoom
	xtile = (lon + 180.0) / 360.0 * n
	ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
	return xtile, ytile


def _contains_keyword(text: str, keyword: str) -> bool:
	if not keyword:
		return False
	return keyword.strip().lower() in text.lower()


def _score_candidate(raw: dict[str, Any], primary_keyword: str) -> float:
	text = str(raw.get("text", ""))
	matched_place = str(raw.get("matched_place", ""))
	display_name = str(raw.get("display_name", ""))
	haystack = f"{text} {matched_place} {display_name}"

	keyword_bonus = 1000.0 if _contains_keyword(haystack, primary_keyword) else 0.0
	matched_place_bonus = 10.0 if matched_place else 0.0
	# Lower idx tends to come from higher OCR confidence (source ordering), so small bonus.
	idx = int(raw.get("idx", 999999))
	idx_bonus = max(0.0, 5.0 - min(5.0, float(idx) * 0.01))
	return keyword_bonus + matched_place_bonus + idx_bonus


def load_primary_coordinate(geocode_json_path: Path, primary_keyword: str) -> CoordinateCandidate:
	payload = json.loads(geocode_json_path.read_text(encoding="utf-8"))
	candidates = payload.get("geocode_candidates", [])
	if not isinstance(candidates, list):
		raise ValueError(f"invalid geocode_candidates format in {geocode_json_path}")

	geocoded: list[dict[str, Any]] = [c for c in candidates if bool(c.get("geocoded", False))]
	if not geocoded:
		raise RuntimeError(
			f"no geocoded candidate found in {geocode_json_path}. Run geocoding first."
		)

	best = max(geocoded, key=lambda c: _score_candidate(c, primary_keyword=primary_keyword))
	return CoordinateCandidate(
		idx=int(best.get("idx", -1)),
		text=str(best.get("text", "")),
		matched_place=str(best.get("matched_place", "")),
		lat=float(best["lat"]),
		lon=float(best["lon"]),
		display_name=str(best.get("display_name", "")),
	)


def download_osm_tile(zoom: int, x: int, y: int, tile_path: Path) -> dict[str, str | int]:
	tile_url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
	request = Request(
		tile_url,
		headers={
			"User-Agent": "capstone-osm-tile-fetcher/1.0",
		},
	)

	tile_path.parent.mkdir(parents=True, exist_ok=True)
	with urlopen(request, timeout=20) as response:
		if response.status != 200:
			raise RuntimeError(f"tile download failed ({response.status}): {tile_url}")
		tile_path.write_bytes(response.read())

	return {
		"zoom": zoom,
		"x": x,
		"y": y,
		"tile_url": tile_url,
		"tile_path": str(tile_path),
	}


def download_osm_tile_image(zoom: int, x: int, y: int) -> tuple[np.ndarray, str]:
	tile_url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
	request = Request(
		tile_url,
		headers={
			"User-Agent": "capstone-osm-tile-fetcher/1.0",
		},
	)

	with urlopen(request, timeout=20) as response:
		if response.status != 200:
			raise RuntimeError(f"tile download failed ({response.status}): {tile_url}")
		data = response.read()

	decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
	if decoded is None:
		raise RuntimeError(f"failed to decode tile image from: {tile_url}")
	return decoded, tile_url


def build_centered_tile_image(
	lat: float,
	lon: float,
	zoom: int,
	tile_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
	x_float, y_float = latlon_to_tile_float(lat=lat, lon=lon, zoom=zoom)
	n = 2**zoom
	native = 256

	global_x = x_float * native
	global_y = y_float * native
	half = tile_size / 2.0
	left = int(math.floor(global_x - half))
	top = int(math.floor(global_y - half))
	right = left + tile_size
	bottom = top + tile_size

	x_start = int(math.floor(left / native))
	x_end = int(math.floor((right - 1) / native))
	y_start = int(math.floor(top / native))
	y_end = int(math.floor((bottom - 1) / native))

	canvas_w = (x_end - x_start + 1) * native
	canvas_h = (y_end - y_start + 1) * native
	canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

	used_tiles: list[dict[str, Any]] = []
	for ty in range(y_start, y_end + 1):
		for tx in range(x_start, x_end + 1):
			if ty < 0 or ty >= n:
				continue
			tx_wrapped = tx % n
			tile_img, tile_url = download_osm_tile_image(zoom=zoom, x=tx_wrapped, y=ty)
			ox = (tx - x_start) * native
			oy = (ty - y_start) * native
			canvas[oy : oy + native, ox : ox + native] = tile_img
			used_tiles.append(
				{
					"x": tx_wrapped,
					"y": ty,
					"tile_url": tile_url,
				}
			)

	canvas_x0 = x_start * native
	canvas_y0 = y_start * native
	crop_x = left - canvas_x0
	crop_y = top - canvas_y0
	centered = canvas[crop_y : crop_y + tile_size, crop_x : crop_x + tile_size].copy()

	meta = {
		"x_float": x_float,
		"y_float": y_float,
		"center_tile_x": int(math.floor(x_float)),
		"center_tile_y": int(math.floor(y_float)),
		"center_pixel_in_output_x": float(global_x - left),
		"center_pixel_in_output_y": float(global_y - top),
		"native_tile_size": native,
		"source_tile_range": {
			"x_start": x_start,
			"x_end": x_end,
			"y_start": y_start,
			"y_end": y_end,
		},
		"source_tiles": used_tiles,
	}
	return centered, meta


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Fetch one primary OSM image centered on the selected geocoding coordinate"
	)
	parser.add_argument(
		"--geocode-json",
		required=True,
		help="Path to *_geocode_candidates.json",
	)
	parser.add_argument(
		"--zoom",
		type=int,
		default=14,
		help="Fixed zoom level (default: 14)",
	)
	parser.add_argument(
		"--primary-keyword",
		default="",
		help="Optional preferred keyword used to pick one geocoded candidate",
	)
	parser.add_argument(
		"--tile-size",
		type=int,
		default=1400,
		help="Saved tile image size in pixels (default: 1400 — matches typical marathon image width for better coverage)",
	)
	parser.add_argument(
		"--output-dir",
		default="./path-to-gpx/output/osm/",
		help="Directory to store OSM tiles and reports",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()

	geocode_json_path = Path(args.geocode_json)
	if not geocode_json_path.exists() or not geocode_json_path.is_file():
		raise FileNotFoundError(f"geocode json not found: {geocode_json_path}")

	zoom = int(args.zoom)
	if not (0 <= zoom <= 19):
		raise ValueError(f"zoom out of range (0..19): {zoom}")
	tile_size = int(args.tile_size)
	if tile_size < 32:
		raise ValueError("tile-size must be >= 32")
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	candidate = load_primary_coordinate(geocode_json_path, primary_keyword=args.primary_keyword)
	print(
		"[INFO] selected coordinate: "
		f"lat={candidate.lat:.7f}, lon={candidate.lon:.7f}, place={candidate.matched_place or candidate.text}"
	)

	report_stem = geocode_json_path.name.replace("_geocode_candidates.json", "")
	if report_stem == geocode_json_path.name:
		report_stem = geocode_json_path.stem

	tiles_dir = output_dir / "tiles"
	tiles_dir.mkdir(parents=True, exist_ok=True)
	centered_image, centered_meta = build_centered_tile_image(
		lat=candidate.lat,
		lon=candidate.lon,
		zoom=zoom,
		tile_size=tile_size,
	)
	x = centered_meta["center_tile_x"]
	y = centered_meta["center_tile_y"]
	tile_path = tiles_dir / f"{report_stem}_z{zoom}_x{x}_y{y}.png"
	if not cv2.imwrite(str(tile_path), centered_image):
		raise RuntimeError(f"failed to save centered OSM image: {tile_path}")

	tile_result = {
		"zoom": zoom,
		"x": x,
		"y": y,
		"tile_path": str(tile_path),
		"saved_width": tile_size,
		"saved_height": tile_size,
		"centered": True,
		"center_pixel_in_output_x": centered_meta["center_pixel_in_output_x"],
		"center_pixel_in_output_y": centered_meta["center_pixel_in_output_y"],
		"source_tile_range": centered_meta["source_tile_range"],
		"source_tiles": centered_meta["source_tiles"],
	}
	print(
		f"[DONE] zoom={zoom} centered_tile=({x},{y}) "
		f"saved={tile_path} size={tile_size}x{tile_size} "
		f"center_px=({centered_meta['center_pixel_in_output_x']:.2f},"
		f"{centered_meta['center_pixel_in_output_y']:.2f})"
	)

	report = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"source_geocode_json": str(geocode_json_path),
		"selected_candidate": asdict(candidate),
		"zoom": zoom,
		"tile_size": tile_size,
		"native_osm_tile_size": 256,
		"primary_keyword": args.primary_keyword,
		"tile_count": 1,
		"tile": tile_result,
	}
	report_path = output_dir / f"{report_stem}_osm_tiles_report.json"
	report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[DONE] osm report: {report_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

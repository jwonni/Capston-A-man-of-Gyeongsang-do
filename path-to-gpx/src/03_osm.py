from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


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


def parse_zoom_levels(zoom_levels_raw: str) -> list[int]:
	zooms: list[int] = []
	for token in zoom_levels_raw.split(","):
		token = token.strip()
		if not token:
			continue
		z = int(token)
		if not (0 <= z <= 19):
			raise ValueError(f"zoom out of range (0..19): {z}")
		zooms.append(z)

	if not zooms:
		raise ValueError("no valid zoom levels")
	return sorted(set(zooms))


def load_best_coordinate(geocode_json_path: Path) -> CoordinateCandidate:
	payload = json.loads(geocode_json_path.read_text(encoding="utf-8"))
	candidates = payload.get("geocode_candidates", [])
	if not isinstance(candidates, list):
		raise ValueError(f"invalid geocode_candidates format in {geocode_json_path}")

	for c in candidates:
		if bool(c.get("geocoded", False)):
			return CoordinateCandidate(
				idx=int(c.get("idx", -1)),
				text=str(c.get("text", "")),
				matched_place=str(c.get("matched_place", "")),
				lat=float(c["lat"]),
				lon=float(c["lon"]),
				display_name=str(c.get("display_name", "")),
			)

	raise RuntimeError(
		f"no geocoded candidate found in {geocode_json_path}. Run geocoding first."
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


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Fetch OSM tiles for the decided geocoding coordinate at multiple zoom levels"
	)
	parser.add_argument(
		"--geocode-json",
		required=True,
		help="Path to *_geocode_candidates.json",
	)
	parser.add_argument(
		"--zoom-levels",
		default="12,14,16",
		help="Comma-separated zoom levels (e.g. 12,14,16)",
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

	zoom_levels = parse_zoom_levels(args.zoom_levels)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	candidate = load_best_coordinate(geocode_json_path)
	print(
		"[INFO] selected coordinate: "
		f"lat={candidate.lat:.7f}, lon={candidate.lon:.7f}, place={candidate.matched_place or candidate.text}"
	)

	report_stem = geocode_json_path.name.replace("_geocode_candidates.json", "")
	if report_stem == geocode_json_path.name:
		report_stem = geocode_json_path.stem

	tiles_dir = output_dir / "tiles"
	tile_results: list[dict[str, str | int]] = []
	for zoom in zoom_levels:
		x, y = latlon_to_tile(candidate.lat, candidate.lon, zoom)
		tile_path = tiles_dir / f"{report_stem}_z{zoom}_x{x}_y{y}.png"
		meta = download_osm_tile(zoom=zoom, x=x, y=y, tile_path=tile_path)
		tile_results.append(meta)
		print(f"[DONE] zoom={zoom} tile=({x},{y}) saved={tile_path}")

	report = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"source_geocode_json": str(geocode_json_path),
		"selected_candidate": asdict(candidate),
		"zoom_levels": zoom_levels,
		"tile_count": len(tile_results),
		"tiles": tile_results,
	}
	report_path = output_dir / f"{report_stem}_osm_tiles_report.json"
	report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[DONE] osm report: {report_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

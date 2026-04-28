'''
지오코딩을 수행하여 OCR 후보 텍스트를 실제 지리적 위치로 변환하는 스크립트
- 입력: OCR 감지 결과 JSON (예: *_ocr_detections.json)
- 출력: 지오코딩 후보 JSON (예: *_geocode_candidates.json)
- 주요 기능:
  - OCR 텍스트에서 GPS 좌표 패턴 추출 (예: "37.5665,126.9780")
  - OCR 텍스트를 지오코딩하여 위도/경도 좌표
- 의존성: geopy (Nominatim 지오코더 사용)
'''
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from geopy.geocoders import Nominatim


# Optional hardcoded credentials for Naver Geocoding.
# Priority order at runtime:
#   1) CLI arguments
#   2) Hardcoded values below
#   3) Environment variables
NAVER_CLIENT_ID_DEFAULT = ""
NAVER_CLIENT_SECRET_DEFAULT = ""


def extract_geo_candidates(text: str) -> dict[str, Any]:
	gps_matches: list[dict[str, Any]] = []

	# Decimal lat,lon pattern, ex: 37.5665,126.9780
	decimal_pattern = re.compile(r"(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")
	for m in decimal_pattern.finditer(text):
		lat = float(m.group(1))
		lon = float(m.group(2))
		if -90 <= lat <= 90 and -180 <= lon <= 180:
			gps_matches.append({"raw": m.group(0), "lat": lat, "lon": lon, "source": "regex_decimal"})

	return {"gps_matches": gps_matches}


def _naver_geocode(query_text: str, client_id: str, client_secret: str) -> dict[str, Any] | None:
	url = (
		"https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
		f"?query={quote(query_text)}&count=1"
	)
	request = Request(
		url,
		headers={
			"X-NCP-APIGW-API-KEY-ID": client_id,
			"X-NCP-APIGW-API-KEY": client_secret,
			"User-Agent": "marathon_gpx_pipeline/1.0",
		},
	)
	with urlopen(request, timeout=15) as response:
		if response.status != 200:
			raise RuntimeError(f"naver geocode http {response.status}")
		payload = json.loads(response.read().decode("utf-8"))

	addresses = payload.get("addresses", [])
	if not addresses:
		return None

	a0 = addresses[0]
	return {
		"lat": float(a0["y"]),
		"lon": float(a0["x"]),
		"display_name": str(a0.get("roadAddress") or a0.get("jibunAddress") or query_text),
	}


def geocode_text_candidates(
	detections: list[dict[str, Any]],
	top_k: int = 5,
	naver_client_id: str = "",
	naver_client_secret: str = "",
) -> list[dict[str, Any]]:
	geolocator = Nominatim(user_agent="ocr_geo_extractor")
	sorted_dets = sorted(
		detections,
		key=lambda x: float(x.get("confidence", 0.0)),
		reverse=True,
	)[:top_k]
	out: list[dict[str, Any]] = []

	for det in sorted_dets:
		text_raw = str(det.get("text", "")).strip()
		matched_place = str(det.get("matched_place", "")).strip()
		query_text = matched_place if matched_place else text_raw
		if len(query_text) < 2:
			continue

		if naver_client_id and naver_client_secret:
			try:
				nav = _naver_geocode(query_text, naver_client_id, naver_client_secret)
				time.sleep(0.15)
			except Exception as exc:
				nav = None
				print(f"[WARN] Naver geocode failed for '{query_text}': {exc}")

			if nav is not None:
				out.append(
					{
						"idx": int(det.get("idx", -1)),
						"text": text_raw,
						"matched_place": matched_place,
						"geocoded": True,
						"lat": nav["lat"],
						"lon": nav["lon"],
						"display_name": nav["display_name"],
						"source": "naver",
					}
				)
				continue

		try:
			location = geolocator.geocode(
				query_text,
				addressdetails=True,
				country_codes="kr",  # restrict to South Korea for accuracy
				language="ko",       # prefer Korean display names
			)
		except Exception as exc:
			out.append(
				{
					"idx": int(det.get("idx", -1)),
					"text": query_text,
					"geocoded": False,
					"reason": str(exc),
				}
			)
			continue

		if location is None:
			out.append(
				{
					"idx": int(det.get("idx", -1)),
					"text": query_text,
					"geocoded": False,
					"reason": "not found",
				}
			)
			continue

		out.append(
			{
				"idx": int(det.get("idx", -1)),
				"text": text_raw,
				"matched_place": matched_place,
				"geocoded": True,
				"lat": float(location.latitude),
				"lon": float(location.longitude),
				"display_name": location.address,
				"source": "nominatim",
			}
		)

	return out


def run_geo_coding(
	ocr_json_path: Path,
	output_dir: Path | None,
	top_k: int,
	naver_client_id: str,
	naver_client_secret: str,
) -> Path:
	data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
	detections = data.get("detections", [])
	if not isinstance(detections, list):
		raise ValueError(f"invalid detections format in: {ocr_json_path}")

	full_text = " ".join(str(d.get("text", "")) for d in detections)
	gps_candidates = extract_geo_candidates(full_text)
	geocode_results = geocode_text_candidates(
		detections,
		top_k=max(1, top_k),
		naver_client_id=naver_client_id,
		naver_client_secret=naver_client_secret,
	)

	image_path = str(data.get("image", ""))
	out_dir = output_dir if output_dir is not None else ocr_json_path.parent
	out_dir.mkdir(parents=True, exist_ok=True)

	stem = ocr_json_path.name
	if stem.endswith("_ocr_detections.json"):
		stem = stem[: -len("_ocr_detections.json")]
	else:
		stem = ocr_json_path.stem

	out_path = out_dir / f"{stem}_geocode_candidates.json"
	out_payload = {
		"image": image_path,
		"mode": data.get("mode", "kr_place_whitelist_only"),
		"source_detection_json": str(ocr_json_path),
		"geocode_candidates": geocode_results,
		"gps_regex_candidates": gps_candidates,
	}
	out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
	return out_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run geocoding from OCR detection JSON generated by ocr.py"
	)
	parser.add_argument(
		"--ocr-json",
		required=True,
		help="Path to *_ocr_detections.json",
	)
	parser.add_argument(
		"--output-dir",
		default="./path-to-gpx/output/geocoding/",
		help="Output directory for geocode JSON",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=5,
		help="Try geocoding for top-k high-confidence texts",
	)
	parser.add_argument(
		"--naver-client-id",
		default="",
		help="Naver API Gateway key ID (default: env NAVER_CLIENT_ID)",
	)
	parser.add_argument(
		"--naver-client-secret",
		default="",
		help="Naver API Gateway key (default: env NAVER_CLIENT_SECRET)",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()

	naver_client_id = str(args.naver_client_id).strip()
	naver_client_secret = str(args.naver_client_secret).strip()
	if not naver_client_id:
		naver_client_id = NAVER_CLIENT_ID_DEFAULT.strip() or os.environ.get("NAVER_CLIENT_ID", "").strip()
	if not naver_client_secret:
		naver_client_secret = NAVER_CLIENT_SECRET_DEFAULT.strip() or os.environ.get("NAVER_CLIENT_SECRET", "").strip()

	ocr_json_path = Path(args.ocr_json)
	if not ocr_json_path.exists() or not ocr_json_path.is_file():
		raise FileNotFoundError(f"ocr detection json not found: {ocr_json_path}")

	out_dir = Path(args.output_dir) if args.output_dir else None
	out_path = run_geo_coding(
		ocr_json_path=ocr_json_path,
		output_dir=out_dir,
		top_k=args.top_k,
		naver_client_id=naver_client_id,
		naver_client_secret=naver_client_secret,
	)
	print(f"[DONE] geocode json: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

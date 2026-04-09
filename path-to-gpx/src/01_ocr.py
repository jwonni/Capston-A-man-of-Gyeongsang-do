from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRDetection:
	idx: int
	text: str
	confidence: float
	matched_place: str
	polygon_xy: list[list[float]]
	center_x: float
	center_y: float
	width: float
	height: float
	norm_center_x: float
	norm_center_y: float


# Hardcoded whitelist: South Korea provinces/cities/landmarks.
KR_PLACE_KEYWORDS = {
	"서울",
	"부산",
	"대구",
	"인천",
	"광주",
	"대전",
	"울산",
	"세종",
	"경기",
	"강원",
	"충북",
	"충남",
	"전북",
	"전남",
	"경북",
	"경남",
	"제주",
	"수원",
	"성남",
	"용인",
	"고양",
	"부천",
	"안양",
	"창원",
	"포항",
	"경주",
	"전주",
	"천안",
	"청주",
	"여의도",
	"한강",
	"한강공원",
	"남산",
	"남산타워",
	"롯데타워",
	"광화문",
	"경복궁",
	"청와대",
	"해운대",
	"광안리",
	"제주공항",
	"김포공항",
	"인천공항",
	"나주",
	"목포",
	"여수",
	"순천",
	"안동",
	"포항",
	"경주",
	"전주",
	"천안",
	"청주",
	"춘천",
	"원주"
}


def normalize_text(s: str) -> str:
	# Keep Korean/English letters and numbers only, then lowercase.
	return re.sub(r"[^0-9a-zA-Z가-힣]", "", s).lower()


def match_kr_place(text: str) -> str:
	norm_text = normalize_text(text)
	if not norm_text:
		return ""

	for keyword in sorted(KR_PLACE_KEYWORDS, key=len, reverse=True):
		norm_key = normalize_text(keyword)
		if norm_key and norm_key in norm_text:
			return keyword
	return ""


def filter_detections_to_kr_places(
	detections: list[OCRDetection], min_confidence: float
) -> tuple[list[OCRDetection], list[dict[str, Any]]]:
	filtered: list[OCRDetection] = []
	rejected: list[dict[str, Any]] = []

	for det in detections:
		matched = match_kr_place(det.text)
		det.matched_place = matched
		if det.confidence >= min_confidence and matched:
			filtered.append(det)
		else:
			rejected.append(
				{
					"idx": det.idx,
					"text": det.text,
					"confidence": det.confidence,
					"reason": "below_confidence_or_not_whitelisted",
				}
			)

	return filtered, rejected


def _try_build_paddle_ocr(lang: str):
	from paddleocr import PaddleOCR

	# Keep compatibility with newer PaddleOCR versions.
	return PaddleOCR(use_textline_orientation=True, lang=lang)


def _run_paddle_ocr(ocr: Any, image_path: str) -> list[OCRDetection]:
	try:
		result = ocr.ocr(image_path, cls=True)
	except Exception:
		# New PaddleOCR API path
		result = ocr.predict(image_path)

	detections: list[OCRDetection] = []
	idx = 1

	if not result:
		return detections

	# Newer API often returns list[dict] with dt_polys/rec_texts/rec_scores.
	if isinstance(result, list) and result and isinstance(result[0], dict):
		for item in result:
			polys = item.get("dt_polys")
			texts = item.get("rec_texts")
			scores = item.get("rec_scores")

			if polys is None or texts is None or scores is None:
				continue

			for poly, text, score in zip(polys, texts, scores):
				polygon = [[float(p[0]), float(p[1])] for p in np.array(poly)]
				detections.append(
					OCRDetection(
						idx=idx,
						text=str(text),
						confidence=float(score),
						matched_place="",
						polygon_xy=polygon,
						center_x=0.0,
						center_y=0.0,
						width=0.0,
						height=0.0,
						norm_center_x=0.0,
						norm_center_y=0.0,
					)
				)
				idx += 1

		return detections

	for line_group in result:
		if not line_group:
			continue
		for item in line_group:
			box = item[0]
			text = item[1][0]
			conf = float(item[1][1])
			polygon = [[float(p[0]), float(p[1])] for p in box]
			detections.append(
				OCRDetection(
					idx=idx,
					text=text,
					confidence=conf,
					matched_place="",
					polygon_xy=polygon,
					center_x=0.0,
					center_y=0.0,
					width=0.0,
					height=0.0,
					norm_center_x=0.0,
					norm_center_y=0.0,
				)
			)
			idx += 1

	return detections


def _run_easyocr(image_path: str, langs: list[str]) -> list[OCRDetection]:
	import easyocr

	reader = easyocr.Reader(langs, gpu=False)
	result = reader.readtext(image_path, detail=1, paragraph=False)
	detections: list[OCRDetection] = []
	idx = 1

	for item in result:
		box = item[0]
		text = item[1]
		conf = float(item[2])
		polygon = [[float(p[0]), float(p[1])] for p in box]
		detections.append(
			OCRDetection(
				idx=idx,
				text=text,
				confidence=conf,
				matched_place="",
				polygon_xy=polygon,
				center_x=0.0,
				center_y=0.0,
				width=0.0,
				height=0.0,
				norm_center_x=0.0,
				norm_center_y=0.0,
			)
		)
		idx += 1

	return detections


def run_ocr(image_path: Path, lang: str = "korean") -> tuple[str, list[OCRDetection]]:
	if lang == "korean":
		paddle_lang = "korean"
		easy_langs = ["ko", "en"]
	elif lang == "english":
		paddle_lang = "en"
		easy_langs = ["en"]
	else:
		paddle_lang = "korean"
		easy_langs = ["ko", "en"]

	try:
		ocr = _try_build_paddle_ocr(paddle_lang)
		detections = _run_paddle_ocr(ocr, str(image_path))
		return "paddleocr", detections
	except Exception as exc:
		print(f"[WARN] PaddleOCR unavailable or failed: {exc}")

	try:
		detections = _run_easyocr(str(image_path), easy_langs)
		return "easyocr", detections
	except Exception as exc:
		raise RuntimeError(
			"No OCR engine available. Install 'paddleocr' (recommended) or 'easyocr'."
		) from exc


def enrich_geometry(detections: list[OCRDetection], width: int, height: int) -> list[OCRDetection]:
	enriched: list[OCRDetection] = []
	for det in detections:
		pts = np.array(det.polygon_xy, dtype=np.float32)
		min_xy = pts.min(axis=0)
		max_xy = pts.max(axis=0)

		box_w = float(max_xy[0] - min_xy[0])
		box_h = float(max_xy[1] - min_xy[1])
		cx = float((min_xy[0] + max_xy[0]) / 2.0)
		cy = float((min_xy[1] + max_xy[1]) / 2.0)

		det.center_x = cx
		det.center_y = cy
		det.width = box_w
		det.height = box_h
		det.norm_center_x = cx / max(width, 1)
		det.norm_center_y = cy / max(height, 1)
		enriched.append(det)
	return enriched


def draw_visualization(image: np.ndarray, detections: list[OCRDetection]) -> np.ndarray:
	vis = image.copy()
	for det in detections:
		pts = np.array(det.polygon_xy, dtype=np.int32)
		cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

		label = f"{det.idx}:{det.text[:24]} ({det.confidence:.2f})"
		tx = int(max(0, det.center_x - 40))
		ty = int(max(20, det.center_y - 8))
		cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	return vis


def save_outputs(
	image_path: Path,
	output_dir: Path,
	engine_name: str,
	detections: list[OCRDetection],
	rejected_detections: list[dict[str, Any]],
	vis_image: np.ndarray,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	stem = image_path.stem
	vis_path = output_dir / f"{stem}_ocr_visualized.png"
	json_path = output_dir / f"{stem}_ocr_detections.json"
	csv_path = output_dir / f"{stem}_ocr_detections.csv"
	rejected_path = output_dir / f"{stem}_ocr_rejected.json"

	cv2.imwrite(str(vis_path), vis_image)

	json_data = {
		"image": str(image_path),
		"engine": engine_name,
		"mode": "kr_place_whitelist_only",
		"whitelist_size": len(KR_PLACE_KEYWORDS),
		"detection_count": len(detections),
		"rejected_count": len(rejected_detections),
		"detections": [
			{
				"idx": d.idx,
				"text": d.text,
				"matched_place": d.matched_place,
				"confidence": d.confidence,
				"polygon_xy": d.polygon_xy,
				"center_x": d.center_x,
				"center_y": d.center_y,
				"width": d.width,
				"height": d.height,
				"norm_center_x": d.norm_center_x,
				"norm_center_y": d.norm_center_y,
			}
			for d in detections
		],
	}
	json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

	with csv_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"idx",
				"text",
				"matched_place",
				"confidence",
				"center_x",
				"center_y",
				"width",
				"height",
				"norm_center_x",
				"norm_center_y",
				"polygon_xy",
			]
		)
		for d in detections:
			writer.writerow(
				[
					d.idx,
					d.text,
					d.matched_place,
					d.confidence,
					d.center_x,
					d.center_y,
					d.width,
					d.height,
					d.norm_center_x,
					d.norm_center_y,
					json.dumps(d.polygon_xy, ensure_ascii=False),
				]
			)

	rejected_path.write_text(json.dumps(rejected_detections, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"[DONE] visualization: {vis_path}")
	print(f"[DONE] detection json: {json_path}")
	print(f"[DONE] detection csv: {csv_path}")
	print(f"[DONE] rejected json: {rejected_path}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Extract text from image, visualize OCR regions, and save geo-ready coordinates."
	)
	parser.add_argument("--image", required=True, help="Input image path")
	parser.add_argument(
		"--output-dir",
		default="./path-to-gpx/output/ocr/",
		help="Output directory for visualization and metadata",
	)
	parser.add_argument(
		"--lang",
		default="korean",
		choices=["korean", "english"],
		help="OCR language preset",
	)
	parser.add_argument(
		"--geocode-top-k",
		type=int,
		default=None,
		help="Deprecated: geocoding moved to geo_coding.py",
	)
	parser.add_argument(
		"--min-confidence",
		type=float,
		default=0.2,
		help="Minimum OCR confidence to keep after KR whitelist filter",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()
	if args.geocode_top_k is not None:
		print("[WARN] '--geocode-top-k' is deprecated in ocr.py. Use geo_coding.py --top-k instead.")

	image_path = Path(args.image)
	if not image_path.exists() or not image_path.is_file():
		raise FileNotFoundError(f"input image not found: {image_path}")

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise RuntimeError(f"failed to load image: {image_path}")

	h, w = image.shape[:2]
	engine_name, detections = run_ocr(image_path=image_path, lang=args.lang)
	detections = enrich_geometry(detections, width=w, height=h)
	detections, rejected = filter_detections_to_kr_places(
		detections=detections,
		min_confidence=max(0.0, min(1.0, args.min_confidence)),
	)
	print(f"[INFO] kept={len(detections)} (KR whitelist), rejected={len(rejected)}")
	if not detections:
		print("[WARN] No whitelist place detected. Try adding keywords in KR_PLACE_KEYWORDS.")
	vis = draw_visualization(image, detections)

	save_outputs(
		image_path=image_path,
		output_dir=Path(args.output_dir),
		engine_name=engine_name,
		detections=detections,
		rejected_detections=rejected,
		vis_image=vis,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

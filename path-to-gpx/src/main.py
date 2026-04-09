from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from geo_coding import run_geo_coding
from ocr import (
	build_arg_parser as _build_ocr_arg_parser,
	draw_visualization,
	enrich_geometry,
	filter_detections_to_kr_places,
	run_ocr,
	save_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
	# Keep argument names aligned with ocr.py for easy migration.
	parser = argparse.ArgumentParser(
		description="Run OCR and geocoding in one pipeline"
	)
	parser.add_argument("--image", required=True, help="Input image path")
	parser.add_argument(
		"--output-dir",
		default="./path-to-gpx/output/ocr/",
		help="Output directory for OCR and geocoding artifacts",
	)
	parser.add_argument(
		"--lang",
		default="korean",
		choices=["korean", "english"],
		help="OCR language preset",
	)
	parser.add_argument(
		"--min-confidence",
		type=float,
		default=0.2,
		help="Minimum OCR confidence to keep after KR whitelist filter",
	)
	parser.add_argument(
		"--geocode-top-k",
		type=int,
		default=5,
		help="Try geocoding for top-k high-confidence texts",
	)
	parser.add_argument(
		"--skip-geocoding",
		action="store_true",
		help="If set, run only OCR and skip geocoding step",
	)
	return parser


def run_pipeline(args: argparse.Namespace) -> int:
	image_path = Path(args.image)
	if not image_path.exists() or not image_path.is_file():
		raise FileNotFoundError(f"input image not found: {image_path}")

	output_dir = Path(args.output_dir)
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
		output_dir=output_dir,
		engine_name=engine_name,
		detections=detections,
		rejected_detections=rejected,
		vis_image=vis,
	)

	if args.skip_geocoding:
		print("[INFO] skip_geocoding enabled; geocoding step skipped.")
		return 0

	ocr_json_path = output_dir / f"{image_path.stem}_ocr_detections.json"
	geo_path = run_geo_coding(
		ocr_json_path=ocr_json_path,
		output_dir=output_dir,
		top_k=max(1, args.geocode_top_k),
	)
	print(f"[DONE] geocode json: {geo_path}")
	return 0


def main() -> int:
	# This call keeps the OCR CLI argument contract discoverable if needed.
	_ = _build_ocr_arg_parser
	parser = build_arg_parser()
	args = parser.parse_args()
	return run_pipeline(args)


if __name__ == "__main__":
	raise SystemExit(main())

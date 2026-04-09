from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
		"--geocoding-output-dir",
		default="./path-to-gpx/output/geocoding/",
		help="Output directory for geocoding artifacts",
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
	src_dir = Path(__file__).resolve().parent
	ocr_script = src_dir / "01_ocr.py"
	geo_script = src_dir / "02_geo_coding.py"

	ocr_cmd = [
		sys.executable,
		str(ocr_script),
		"--image",
		str(image_path),
		"--output-dir",
		str(output_dir),
		"--lang",
		args.lang,
		"--min-confidence",
		str(args.min_confidence),
	]
	print("[INFO] Running OCR step...")
	subprocess.run(ocr_cmd, check=True)

	if args.skip_geocoding:
		print("[INFO] skip_geocoding enabled; geocoding step skipped.")
		return 0

	ocr_json_path = output_dir / f"{image_path.stem}_ocr_detections.json"
	geocoding_output_dir = Path(args.geocoding_output_dir)
	geo_cmd = [
		sys.executable,
		str(geo_script),
		"--ocr-json",
		str(ocr_json_path),
		"--output-dir",
		str(geocoding_output_dir),
		"--top-k",
		str(max(1, args.geocode_top_k)),
	]
	print("[INFO] Running geocoding step...")
	subprocess.run(geo_cmd, check=True)
	return 0


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()
	return run_pipeline(args)


if __name__ == "__main__":
	raise SystemExit(main())

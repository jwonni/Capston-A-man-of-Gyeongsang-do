from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MatchingStats:
	detector: str
	keypoints_marathon: int
	keypoints_map: int
	raw_matches: int
	good_matches: int
	inlier_matches: int
	inlier_ratio: float
	homography_found: bool


def load_color_and_gray(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise RuntimeError(f"failed to load image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image, gray


def preprocess(gray: np.ndarray) -> np.ndarray:
	# Local contrast boost helps matching across differently rendered maps/images.
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	return clahe.apply(gray)


def build_detector(name: str):
	name = name.lower()
	if name == "sift":
		if hasattr(cv2, "SIFT_create"):
			return cv2.SIFT_create(), cv2.NORM_L2, "sift"
		print("[WARN] SIFT is unavailable in this OpenCV build. Falling back to ORB.")
		return cv2.ORB_create(nfeatures=6000, fastThreshold=5), cv2.NORM_HAMMING, "orb"
	if name == "orb":
		return cv2.ORB_create(nfeatures=6000, fastThreshold=5), cv2.NORM_HAMMING, "orb"
	raise ValueError(f"unsupported detector: {name}")


def match_descriptors(
	des1: np.ndarray,
	des2: np.ndarray,
	norm_type: int,
	ratio_test: float,
) -> list[cv2.DMatch]:
	matcher = cv2.BFMatcher(norm_type)
	raw_knn = matcher.knnMatch(des1, des2, k=2)
	good: list[cv2.DMatch] = []
	for pair in raw_knn:
		if len(pair) < 2:
			continue
		m, n = pair
		if m.distance < ratio_test * n.distance:
			good.append(m)
	good.sort(key=lambda x: x.distance)
	return good


def estimate_homography(
	kp1: list[cv2.KeyPoint],
	kp2: list[cv2.KeyPoint],
	good_matches: list[cv2.DMatch],
	ransac_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
	if len(good_matches) < 4:
		return None, None

	src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
	return h_mat, mask


def make_overlay(
	marathon_color: np.ndarray,
	map_color: np.ndarray,
	h_mat: np.ndarray,
	alpha: float,
) -> np.ndarray:
	h_map, w_map = map_color.shape[:2]
	warped = cv2.warpPerspective(marathon_color, h_mat, (w_map, h_map))
	overlay = cv2.addWeighted(warped, alpha, map_color, 1.0 - alpha, 0.0)
	return overlay


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Feature matching between marathon image and OSM map, with homography estimation"
	)
	parser.add_argument("--marathon-image", required=True, help="Path to marathon image")
	parser.add_argument("--map-image", required=True, help="Path to OSM map image")
	parser.add_argument(
		"--detector",
		default="sift",
		choices=["sift", "orb"],
		help="Feature detector type",
	)
	parser.add_argument(
		"--ratio-test",
		type=float,
		default=0.75,
		help="Lowe ratio test threshold",
	)
	parser.add_argument(
		"--ransac-threshold",
		type=float,
		default=5.0,
		help="RANSAC reprojection threshold for homography",
	)
	parser.add_argument(
		"--overlay-alpha",
		type=float,
		default=0.5,
		help="Blending alpha for warped marathon image over map image",
	)
	parser.add_argument(
		"--output-dir",
		default="./path-to-gpx/output/homography/",
		help="Directory to save homography outputs",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()

	marathon_image_path = Path(args.marathon_image)
	map_image_path = Path(args.map_image)
	if not marathon_image_path.exists() or not marathon_image_path.is_file():
		raise FileNotFoundError(f"marathon image not found: {marathon_image_path}")
	if not map_image_path.exists() or not map_image_path.is_file():
		raise FileNotFoundError(f"map image not found: {map_image_path}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"[INFO] loading marathon image: {marathon_image_path}")
	marathon_color, marathon_gray = load_color_and_gray(marathon_image_path)
	print(f"[INFO] loading map image: {map_image_path}")
	map_color, map_gray = load_color_and_gray(map_image_path)

	marathon_gray = preprocess(marathon_gray)
	map_gray = preprocess(map_gray)

	detector, norm_type, detector_used = build_detector(args.detector)
	kp1, des1 = detector.detectAndCompute(marathon_gray, None)
	kp2, des2 = detector.detectAndCompute(map_gray, None)

	if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
		raise RuntimeError("no descriptors found; try different images or detector")

	good_matches = match_descriptors(des1, des2, norm_type=norm_type, ratio_test=args.ratio_test)
	h_mat, inlier_mask = estimate_homography(
		kp1,
		kp2,
		good_matches,
		ransac_threshold=args.ransac_threshold,
	)

	inlier_matches: list[cv2.DMatch] = []
	if inlier_mask is not None:
		for m, flag in zip(good_matches, inlier_mask.ravel().tolist()):
			if int(flag) == 1:
				inlier_matches.append(m)

	stats = MatchingStats(
		detector=detector_used,
		keypoints_marathon=len(kp1),
		keypoints_map=len(kp2),
		raw_matches=min(len(des1), len(des2)),
		good_matches=len(good_matches),
		inlier_matches=len(inlier_matches),
		inlier_ratio=(len(inlier_matches) / max(1, len(good_matches))),
		homography_found=(h_mat is not None),
	)

	stem = f"{marathon_image_path.stem}_to_{map_image_path.stem}"
	matches_path = output_dir / f"{stem}_matches.png"
	inliers_path = output_dir / f"{stem}_inliers.png"
	overlay_path = output_dir / f"{stem}_overlay.png"
	report_path = output_dir / f"{stem}_report.json"

	matches_vis = cv2.drawMatches(
		marathon_color,
		kp1,
		map_color,
		kp2,
		good_matches[:200],
		None,
		flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)
	cv2.imwrite(str(matches_path), matches_vis)
	print(f"[DONE] matches image: {matches_path}")

	inlier_draw = inlier_matches if inlier_matches else good_matches[:80]
	inliers_vis = cv2.drawMatches(
		marathon_color,
		kp1,
		map_color,
		kp2,
		inlier_draw,
		None,
		flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)
	cv2.imwrite(str(inliers_path), inliers_vis)
	print(f"[DONE] inlier image: {inliers_path}")

	if h_mat is not None:
		overlay = make_overlay(
			marathon_color=marathon_color,
			map_color=map_color,
			h_mat=h_mat,
			alpha=max(0.0, min(1.0, args.overlay_alpha)),
		)
		cv2.imwrite(str(overlay_path), overlay)
		print(f"[DONE] overlay image: {overlay_path}")
	else:
		print("[WARN] homography could not be estimated (not enough robust matches).")

	report = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"input": {
			"marathon_image": str(marathon_image_path),
			"map_image": str(map_image_path),
		},
		"params": {
			"detector_requested": args.detector,
			"detector_used": detector_used,
			"ratio_test": args.ratio_test,
			"ransac_threshold": args.ransac_threshold,
			"overlay_alpha": args.overlay_alpha,
		},
		"stats": asdict(stats),
		"homography_matrix": h_mat.tolist() if h_mat is not None else None,
		"outputs": {
			"matches_image": str(matches_path),
			"inliers_image": str(inliers_path),
			"overlay_image": str(overlay_path) if h_mat is not None else None,
		},
	}
	report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[DONE] report json: {report_path}")

	if h_mat is None:
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

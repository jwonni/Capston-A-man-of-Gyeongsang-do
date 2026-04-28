from __future__ import annotations

import argparse
import json
import math
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
    homography_valid: bool


def load_color_and_gray(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def load_route_mask(mask_path: Path, target_hw: tuple[int, int], dilation_px: int = 40) -> np.ndarray | None:
    """Load binary route mask, resize to target, dilate to give SIFT room around route pixels.

    target_hw: (height, width) of the resized marathon image.
    Returns uint8 mask (255 = detect here, 0 = skip).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN] failed to read mask: {mask_path}")
        return None
    h_tgt, w_tgt = target_hw
    if mask.shape[:2] != (h_tgt, w_tgt):
        mask = cv2.resize(mask, (w_tgt, h_tgt), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
        binary = cv2.dilate(binary, kernel)
    return binary


def preprocess(gray: np.ndarray, use_edges: bool = True) -> np.ndarray:
    """CLAHE + optional Canny edge detection.

    Edge detection drastically reduces the visual domain gap between a stylized
    marathon route image and an OSM tile — both become structural line drawings.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if use_edges:
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(edges, kernel, iterations=1)
    return enhanced


def resize_to_height(img: np.ndarray, target_h: int) -> tuple[np.ndarray, float]:
    """Resize image so its height equals target_h; return (resized, scale_factor)."""
    h, w = img.shape[:2]
    if h == target_h:
        return img, 1.0
    scale = target_h / h
    new_w = int(round(w * scale))
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, target_h), interpolation=interp), scale


def build_detector(name: str):
    name = name.lower()
    if name == "sift":
        if hasattr(cv2, "SIFT_create"):
            return (
                cv2.SIFT_create(
                    nfeatures=0,
                    nOctaveLayers=5,
                    contrastThreshold=0.02,   # lower → more features on faint route lines
                    edgeThreshold=10,
                    sigma=1.6,
                ),
                cv2.NORM_L2,
                "sift",
            )
        print("[WARN] SIFT unavailable, falling back to ORB.")
        return cv2.ORB_create(nfeatures=8000, fastThreshold=5), cv2.NORM_HAMMING, "orb"
    if name == "orb":
        return cv2.ORB_create(nfeatures=8000, fastThreshold=5), cv2.NORM_HAMMING, "orb"
    if name == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING, "akaze"
    raise ValueError(f"unsupported detector: {name}")


def match_descriptors(
    des1: np.ndarray,
    des2: np.ndarray,
    norm_type: int,
    ratio_test: float,
) -> tuple[list[cv2.DMatch], int]:
    """Run BF kNN + Lowe ratio test. Returns (good_matches, raw_knn_count)."""
    matcher = cv2.BFMatcher(norm_type)
    raw_knn = matcher.knnMatch(des1, des2, k=2)
    raw_count = len(raw_knn)
    good: list[cv2.DMatch] = []
    for pair in raw_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    good.sort(key=lambda x: x.distance)
    return good, raw_count


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


def validate_homography(
    h_mat: np.ndarray | None,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> tuple[bool, str]:
    """Return (is_valid, reason_str). Rejects degenerate homographies.

    Checks:
      1. Determinant of the 2×2 linear part is non-trivial (not collapsing to a line/point).
      2. Scale factor is within a sane range.
      3. Projected source corners overlap the destination image.
      4. Projected convex hull area is large enough to be meaningful.
    """
    if h_mat is None:
        return False, "no matrix"

    det = float(h_mat[0, 0] * h_mat[1, 1] - h_mat[0, 1] * h_mat[1, 0])
    if abs(det) < 0.01:
        return False, f"degenerate linear part (det={det:.6f} ≈ 0)"

    scale = math.sqrt(abs(det))
    if scale < 0.05 or scale > 50.0:
        return False, f"unreasonable scale factor ({scale:.3f})"

    h_src, w_src = src_hw
    h_dst, w_dst = dst_hw
    corners_src = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]).reshape(-1, 1, 2)
    try:
        corners_dst = cv2.perspectiveTransform(corners_src, h_mat)
    except cv2.error as exc:
        return False, f"perspectiveTransform failed: {exc}"

    xs = corners_dst[:, 0, 0]
    ys = corners_dst[:, 0, 1]
    if xs.max() < 0 or xs.min() > w_dst or ys.max() < 0 or ys.min() > h_dst:
        return False, "projected corners lie entirely outside destination image"

    hull_area = float(cv2.contourArea(cv2.convexHull(corners_dst)))
    if hull_area < 100.0:
        return False, f"projected area too small ({hull_area:.1f} px²)"

    return True, "ok"


def scale_homography_to_original(h_resized: np.ndarray, scale_factor: float) -> np.ndarray:
    """Convert homography from resized-marathon-space to original-marathon-space.

    If p_resized = scale * p_orig, then:
        H_resized @ p_resized ≈ q_map
        H_resized @ S @ p_orig ≈ q_map
    so H_original = H_resized @ S, where S = diag(scale, scale, 1).
    """
    S = np.array(
        [[scale_factor, 0.0, 0.0],
         [0.0, scale_factor, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return h_resized @ S


def make_overlay(
    marathon_color: np.ndarray,
    map_color: np.ndarray,
    h_mat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    h_map, w_map = map_color.shape[:2]
    warped = cv2.warpPerspective(marathon_color, h_mat, (w_map, h_map))
    return cv2.addWeighted(warped, alpha, map_color, 1.0 - alpha, 0.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature matching between marathon image and OSM map tile, "
            "with optional route-mask guidance and homography estimation."
        )
    )
    parser.add_argument("--marathon-image", required=True, help="Path to marathon image")
    parser.add_argument("--map-image", required=True, help="Path to OSM map tile image")
    parser.add_argument(
        "--mask-image",
        default=None,
        help=(
            "Optional path to binary route mask (white route on black background). "
            "Restricts SIFT keypoint detection to the route area, dramatically "
            "reducing false matches from text/logos/decorations."
        ),
    )
    parser.add_argument(
        "--detector",
        default="sift",
        choices=["sift", "orb", "akaze"],
        help="Feature detector/descriptor type (default: sift)",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        default=False,
        help="Disable Canny edge preprocessing (enabled by default). "
             "Edge preprocessing bridges the visual domain gap between marathon and OSM images.",
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.75,
        help="Lowe ratio test threshold (default: 0.75)",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in pixels (default: 5.0)",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=40,
        help="Dilation radius in pixels applied to route mask before using it as SIFT mask (default: 40)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Blending alpha for the warped marathon image over the map image (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="./path-to-gpx/output/homography/",
        help="Directory to save outputs",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    marathon_image_path = Path(args.marathon_image)
    map_image_path = Path(args.map_image)
    if not marathon_image_path.exists():
        raise FileNotFoundError(f"marathon image not found: {marathon_image_path}")
    if not map_image_path.exists():
        raise FileNotFoundError(f"map image not found: {map_image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_edges = not args.no_edges

    print(f"[INFO] loading marathon image: {marathon_image_path}")
    marathon_color, marathon_gray = load_color_and_gray(marathon_image_path)
    print(f"[INFO] loading map image:      {map_image_path}")
    map_color, map_gray = load_color_and_gray(map_image_path)

    h_marathon, w_marathon = marathon_gray.shape[:2]
    h_map, w_map = map_gray.shape[:2]
    print(f"[INFO] marathon size: {w_marathon}×{h_marathon},  map size: {w_map}×{h_map}")

    # --- Scale normalization ---
    # Resize marathon image to the same height as the map tile so that SIFT
    # finds keypoints at matching scales.  The homography is later converted
    # back to original-image space for the overlay / GPX pipeline.
    marathon_color_resized, scale_factor = resize_to_height(marathon_color, h_map)
    marathon_gray_resized, _ = resize_to_height(marathon_gray, h_map)
    if abs(scale_factor - 1.0) > 0.01:
        print(
            f"[INFO] resized marathon by {scale_factor:.3f}× → "
            f"{marathon_color_resized.shape[1]}×{marathon_color_resized.shape[0]}"
        )

    # --- Route mask (optional but strongly recommended) ---
    # The mask restricts SIFT to keypoints along the marathon route, which
    # corresponds to the road network visible in the OSM tile.
    mask_for_sift: np.ndarray | None = None
    if args.mask_image:
        mask_path = Path(args.mask_image)
        if mask_path.exists():
            mask_for_sift = load_route_mask(
                mask_path,
                target_hw=marathon_gray_resized.shape[:2],
                dilation_px=args.mask_dilation,
            )
            if mask_for_sift is not None:
                route_px = int(np.count_nonzero(mask_for_sift))
                total_px = mask_for_sift.size
                print(
                    f"[INFO] route mask loaded: {mask_path.name}  "
                    f"active region {route_px}/{total_px} px "
                    f"({100.0*route_px/total_px:.1f}%)"
                )
        else:
            print(f"[WARN] mask image not found: {mask_path}")

    # --- Preprocessing ---
    print(f"[INFO] preprocessing  use_edges={use_edges}")
    marathon_proc = preprocess(marathon_gray_resized, use_edges=use_edges)
    map_proc = preprocess(map_gray, use_edges=use_edges)

    # --- Feature detection & description ---
    detector, norm_type, detector_used = build_detector(args.detector)
    kp1, des1 = detector.detectAndCompute(marathon_proc, mask_for_sift)
    kp2, des2 = detector.detectAndCompute(map_proc, None)
    print(f"[INFO] keypoints  marathon={len(kp1)}  map={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print("[ERROR] too few descriptors — cannot match. Try without --no-edges or provide --mask-image.")
        return 2

    # --- Matching ---
    good_matches, raw_count = match_descriptors(
        des1, des2, norm_type=norm_type, ratio_test=args.ratio_test
    )
    print(
        f"[INFO] matching  raw_knn={raw_count}  good={len(good_matches)}"
        f"  (ratio_test={args.ratio_test})"
    )

    h_mat, inlier_mask = estimate_homography(
        kp1, kp2, good_matches, ransac_threshold=args.ransac_threshold
    )

    inlier_matches: list[cv2.DMatch] = []
    if inlier_mask is not None:
        for m, flag in zip(good_matches, inlier_mask.ravel().tolist()):
            if int(flag) == 1:
                inlier_matches.append(m)

    # --- Homography validation ---
    is_valid, validity_reason = validate_homography(
        h_mat,
        src_hw=marathon_gray_resized.shape[:2],
        dst_hw=map_gray.shape[:2],
    )
    if h_mat is not None and not is_valid:
        print(f"[WARN] homography is degenerate/invalid: {validity_reason}")

    # Convert to original-image space only if the homography is valid
    h_mat_original: np.ndarray | None = None
    if h_mat is not None and is_valid:
        if abs(scale_factor - 1.0) > 0.01:
            h_mat_original = scale_homography_to_original(h_mat, scale_factor)
        else:
            h_mat_original = h_mat
    elif h_mat is not None and not is_valid:
        h_mat_original = None   # discard degenerate result

    stats = MatchingStats(
        detector=detector_used,
        keypoints_marathon=len(kp1),
        keypoints_map=len(kp2),
        raw_matches=raw_count,           # fixed: was min(len(des1),len(des2))
        good_matches=len(good_matches),
        inlier_matches=len(inlier_matches),
        inlier_ratio=(len(inlier_matches) / max(1, len(good_matches))),
        homography_found=(h_mat is not None),
        homography_valid=is_valid,
    )

    stem = f"{marathon_image_path.stem}_to_{map_image_path.stem}"
    matches_path = output_dir / f"{stem}_matches.png"
    inliers_path = output_dir / f"{stem}_inliers.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    report_path = output_dir / f"{stem}_report.json"

    # Draw on resized marathon for consistent visualisation
    matches_vis = cv2.drawMatches(
        marathon_color_resized, kp1,
        map_color, kp2,
        good_matches[:200], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(matches_path), matches_vis)
    print(f"[DONE] matches image: {matches_path}")

    inlier_draw = inlier_matches if inlier_matches else good_matches[:80]
    inliers_vis = cv2.drawMatches(
        marathon_color_resized, kp1,
        map_color, kp2,
        inlier_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(inliers_path), inliers_vis)
    print(f"[DONE] inlier image: {inliers_path}")

    if h_mat_original is not None:
        overlay = make_overlay(
            marathon_color=marathon_color,   # use original (un-resized)
            map_color=map_color,
            h_mat=h_mat_original,
            alpha=max(0.0, min(1.0, args.overlay_alpha)),
        )
        cv2.imwrite(str(overlay_path), overlay)
        print(f"[DONE] overlay image: {overlay_path}")
    else:
        print("[WARN] no valid homography — overlay not produced.")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "marathon_image": str(marathon_image_path),
            "map_image": str(map_image_path),
            "mask_image": str(args.mask_image) if args.mask_image else None,
        },
        "params": {
            "detector_requested": args.detector,
            "detector_used": detector_used,
            "use_edges": use_edges,
            "mask_dilation_px": args.mask_dilation,
            "ratio_test": args.ratio_test,
            "ransac_threshold": args.ransac_threshold,
            "overlay_alpha": args.overlay_alpha,
            "marathon_resize_scale": scale_factor,
        },
        "stats": asdict(stats),
        "homography_validity": {"valid": is_valid, "reason": validity_reason},
        "homography_matrix": h_mat_original.tolist() if h_mat_original is not None else None,
        "outputs": {
            "matches_image": str(matches_path),
            "inliers_image": str(inliers_path),
            "overlay_image": str(overlay_path) if h_mat_original is not None else None,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report json: {report_path}")

    print(
        f"\n[SUMMARY] detector={detector_used}  kp={len(kp1)}/{len(kp2)}"
        f"  good={len(good_matches)}  inliers={len(inlier_matches)}"
        f"  ratio={stats.inlier_ratio:.1%}  valid={is_valid}  reason={validity_reason}"
    )

    return 0 if h_mat_original is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

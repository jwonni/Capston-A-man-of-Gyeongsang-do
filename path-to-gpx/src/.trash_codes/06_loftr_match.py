"""06_loftr_match.py — Cross-Domain Feature Matching (LoFTR + MAGSAC++)

For each source-patch / reference-tile pair produced by 05_anchor_patches.py:
  1. Run LoFTR (kornia) for dense, cross-domain correspondences.
     Falls back to SIFT + ratio-test if kornia is not installed.
  2. Apply MAGSAC++ (cv2.USAC_MAGSAC, falls back to RANSAC) to reject
     outliers and estimate a local homography per anchor.
  3. Convert inlier tile-pixel correspondences to (lat, lon) via the stored
     geotransform from 05_anchor_patches.py.

Output is a list of Image-A pixel ↔ (lat, lon) control points, one set per
anchor, which feed directly into 07_tps_align.py.

Inputs
------
--anchors-json   Output of 05_anchor_patches.py
--confidence     LoFTR confidence threshold (default: 0.2)
--magsac-thr     MAGSAC++ reprojection threshold in pixels (default: 3.0)
--output-dir     Directory to write output (default: ./path-to-gpx/output/loftr/)

Output
------
<output-dir>/<stem>_loftr_matches.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geotransform helpers (mirror 05_anchor_patches.py)
# ---------------------------------------------------------------------------

def _tile_float_to_latlon(tx: float, ty: float, zoom: int) -> tuple[float, float]:
    n = 1 << zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lat, lon


def tile_px_to_latlon(px: float, py: float, gt: dict) -> tuple[float, float]:
    """Convert a pixel (px, py) in a reference tile image to (lat, lon)."""
    tx_f = gt["top_left_tx_float"] + px / gt["osm_tile_px"]
    ty_f = gt["top_left_ty_float"] + py / gt["osm_tile_px"]
    return _tile_float_to_latlon(tx_f, ty_f, gt["zoom"])


# ---------------------------------------------------------------------------
# LoFTR matcher
# ---------------------------------------------------------------------------

def _match_loftr(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match img0 (source patch) vs img1 (OSM reference tile) with kornia LoFTR.

    Returns (mkpts0, mkpts1, conf) — float32 arrays of shape (N, 2) and (N,).
    Raises ImportError if kornia is not installed.
    """
    import torch
    import kornia.feature as KF

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load once per process (cached by Python's module system after first call)
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def _to_tensor(bgr: np.ndarray) -> "torch.Tensor":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # LoFTR requires dims divisible by 8
        h8 = (h // 8) * 8 or 8
        w8 = (w // 8) * 8 or 8
        if (h8, w8) != (h, w):
            gray = cv2.resize(gray, (w8, h8), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t.to(device)

    with torch.inference_mode():
        out = matcher({"image0": _to_tensor(img0_bgr), "image1": _to_tensor(img1_bgr)})

    mkpts0 = out["keypoints0"].cpu().numpy().astype(np.float32)
    mkpts1 = out["keypoints1"].cpu().numpy().astype(np.float32)
    conf   = out["confidence"].cpu().numpy().astype(np.float32)

    keep = conf >= conf_thr
    return mkpts0[keep], mkpts1[keep], conf[keep]


# ---------------------------------------------------------------------------
# SIFT fallback (for when kornia is not available)
# ---------------------------------------------------------------------------

def _match_sift(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIFT + CLAHE-edges + BF k-NN + Lowe ratio-test fallback."""

    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        return cv2.dilate(edges, np.ones((2, 2), np.uint8))

    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
    kp0, des0 = sift.detectAndCompute(_preprocess(img0_bgr), None)
    kp1, des1 = sift.detectAndCompute(_preprocess(img1_bgr), None)

    empty = (np.zeros((0, 2), np.float32),) * 2 + (np.zeros(0, np.float32),)
    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return empty

    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des0, des1, k=2)
    pts0, pts1 = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                pts0.append(kp0[m.queryIdx].pt)
                pts1.append(kp1[m.trainIdx].pt)

    if not pts0:
        return empty
    a0 = np.array(pts0, dtype=np.float32)
    a1 = np.array(pts1, dtype=np.float32)
    return a0, a1, np.ones(len(pts0), dtype=np.float32)


# ---------------------------------------------------------------------------
# MAGSAC++ homography estimation
# ---------------------------------------------------------------------------

# cv2.USAC_MAGSAC = 8 (available in OpenCV >= 4.7; falls back gracefully)
_MAGSAC = getattr(cv2, "USAC_MAGSAC", None)


def _find_homography_robust(
    pts0: np.ndarray,
    pts1: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate homography with MAGSAC++ (or RANSAC if method unavailable)."""
    if len(pts0) < 4:
        return None, None

    p0 = pts0.reshape(-1, 1, 2).astype(np.float64)
    p1 = pts1.reshape(-1, 1, 2).astype(np.float64)

    if _MAGSAC is not None:
        try:
            H, mask = cv2.findHomography(
                p0, p1, _MAGSAC, threshold, confidence=0.999, maxIters=10_000
            )
            if H is not None:
                return H, mask
        except cv2.error:
            pass  # fall through to RANSAC

    # RANSAC fallback
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, threshold)
    return H, mask


# ---------------------------------------------------------------------------
# Per-anchor processing
# ---------------------------------------------------------------------------

def _process_anchor(
    anchor: dict,
    conf_thr: float,
    magsac_thr: float,
) -> dict[str, Any]:
    idx = anchor["idx"]
    src_path = Path(anchor["src_patch"])
    ref_path = Path(anchor["ref_tile"])
    gt = anchor["geotransform"]
    off = anchor["src_offset"]

    base = {"idx": idx, "text": anchor.get("text", ""), "control_points": []}

    for p, label in [(src_path, "src_patch"), (ref_path, "ref_tile")]:
        if not p.exists():
            return {**base, "status": f"missing_{label}"}

    img0 = cv2.imread(str(src_path))
    img1 = cv2.imread(str(ref_path))
    if img0 is None or img1 is None:
        return {**base, "status": "load_error"}

    # ---- Feature matching ------------------------------------------------
    matcher_used = "loftr"
    try:
        mkpts0, mkpts1, conf = _match_loftr(img0, img1, conf_thr)
    except ImportError:
        print("    [WARN] kornia not installed — falling back to SIFT.")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"
    except Exception as exc:
        print(f"    [WARN] LoFTR error ({exc}) — falling back to SIFT.")
        mkpts0, mkpts1, conf = _match_sift(img0, img1)
        matcher_used = "sift_fallback"

    raw_n = len(mkpts0)
    print(f"    raw matches={raw_n}  matcher={matcher_used}")

    if raw_n < 4:
        return {**base, "status": "too_few_matches",
                "raw_matches": raw_n, "matcher_used": matcher_used}

    # ---- MAGSAC++ / RANSAC -----------------------------------------------
    H, mask = _find_homography_robust(mkpts0, mkpts1, magsac_thr)
    inlier_n = int(mask.ravel().astype(bool).sum()) if mask is not None else 0
    print(f"    inliers={inlier_n}/{raw_n}  H={'found' if H is not None else 'none'}")

    if H is None or inlier_n < 4:
        return {**base, "status": "homography_failed",
                "raw_matches": raw_n, "matcher_used": matcher_used,
                "inlier_matches": inlier_n}

    # ---- Convert inlier matches to Image-A pixel ↔ lat/lon ---------------
    inlier_mask = mask.ravel().astype(bool)
    pts0_in = mkpts0[inlier_mask]
    pts1_in = mkpts1[inlier_mask]

    off_x = float(off["offset_x"])
    off_y = float(off["offset_y"])

    control_points: list[dict] = []
    for (px_p, py_p), (tx_t, ty_t) in zip(pts0_in, pts1_in):
        lat, lon = tile_px_to_latlon(float(tx_t), float(ty_t), gt)
        control_points.append({
            "img_x":   float(px_p) + off_x,
            "img_y":   float(py_p) + off_y,
            "lat":     lat,
            "lon":     lon,
            "patch_x": float(px_p),
            "patch_y": float(py_p),
            "tile_x":  float(tx_t),
            "tile_y":  float(ty_t),
        })

    return {
        **base,
        "status":        "ok",
        "matcher_used":  matcher_used,
        "raw_matches":   raw_n,
        "inlier_matches": inlier_n,
        "inlier_ratio":  inlier_n / raw_n,
        "control_points": control_points,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-domain feature matching: LoFTR + MAGSAC++ per anchor patch pair."
    )
    p.add_argument("--anchors-json", required=True,
                   help="Anchors JSON from 05_anchor_patches.py")
    p.add_argument("--confidence", type=float, default=0.2,
                   help="LoFTR confidence threshold (default: 0.2)")
    p.add_argument("--magsac-thr", type=float, default=3.0,
                   help="MAGSAC++ reprojection threshold in pixels (default: 3.0)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/loftr/",
                   help="Output directory (default: ./path-to-gpx/output/loftr/)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    anchors_path = Path(args.anchors_json)
    if not anchors_path.exists():
        print(f"[ERROR] anchors JSON not found: {anchors_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data    = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = data["anchors"]
    stem    = Path(data["image"]).stem

    print(f"[INFO] processing {len(anchors)} anchor pair(s) from {anchors_path.name}")
    if _MAGSAC is not None:
        print("[INFO] MAGSAC++ available (cv2.USAC_MAGSAC)")
    else:
        print("[INFO] cv2.USAC_MAGSAC not found — will use RANSAC as fallback")

    results: list[dict] = []
    total_cps = 0

    for i, anchor in enumerate(anchors):
        print(f"\n[{i+1}/{len(anchors)}] anchor idx={anchor['idx']}  '{anchor.get('text', '')}'")
        res = _process_anchor(anchor, args.confidence, args.magsac_thr)
        results.append(res)
        n_cps = len(res.get("control_points", []))
        total_cps += n_cps
        print(f"    → status={res['status']}  control_points={n_cps}")

    out_json = output_dir / f"{stem}_loftr_matches.json"
    out_json.write_text(json.dumps({
        "anchors_json":          str(anchors_path),
        "image":                 data["image"],
        "image_width":           data["image_width"],
        "image_height":          data["image_height"],
        "confidence_threshold":  args.confidence,
        "magsac_threshold":      args.magsac_thr,
        "total_control_points":  total_cps,
        "anchor_results":        results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] {total_cps} control point(s) across {len(anchors)} anchor(s) → {out_json}")
    return 0 if total_cps > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

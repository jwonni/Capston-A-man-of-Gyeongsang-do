"""08_path_extract.py — Orange Path Extraction + Skeletonization

Isolates the orange dashed marathon route from the schematic image,
reduces it to a 1-pixel centreline, orders the centreline pixels into a
continuous traversal sequence, and subsamples to a target waypoint count.

Algorithm
---------
1. HSV colour thresholding — two bands cover orange (≈H 5-25) and
   orange-red wrap-around (≈H 160-180).  Thresholds are CLI-configurable.
2. Morphological closing — bridges the dashes of a dashed line.
3. Morphological opening — removes tiny noise specks.
4. Largest connected component — discards arrows, labels, decorations.
5. Zhang-Suen skeletonization (skimage; falls back to iterative erosion).
6. 8-connected path ordering — greedy walk from an endpoint; KDTree
   nearest-neighbour gap bridging at dead-ends.
7. Uniform subsampling to ``--subsample`` waypoints.

Inputs
------
--image      Marathon schematic image (Image A)
--hue-lo1    Lower hue bound, primary orange band  [0–180] (default: 5)
--hue-hi1    Upper hue bound, primary orange band           (default: 25)
--hue-lo2    Lower hue bound, red-orange wrap-around        (default: 160)
--hue-hi2    Upper hue bound, red-orange wrap-around        (default: 180)
--sat-lo     Minimum saturation                             (default: 100)
--val-lo     Minimum brightness value                       (default: 80)
--close-px   Closing-kernel radius in px (bridges dashes)   (default: 15)
--subsample  Target waypoint count after ordering           (default: 500)
--output-dir Output directory

Outputs
-------
<output-dir>/<stem>_orange_mask.png      — binary segmentation result
<output-dir>/<stem>_skeleton.png         — 1-px centreline
<output-dir>/<stem>_skeleton_path.json   — ordered {px, py} waypoints
<output-dir>/<stem>_path_visualized.png  — extracted path overlaid on image
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_orange(
    bgr: np.ndarray,
    hue_lo1: int, hue_hi1: int,
    hue_lo2: int, hue_hi2: int,
    sat_lo: int,
    val_lo: int,
) -> np.ndarray:
    """
    Return a uint8 binary mask of orange pixels using two HSV hue bands.

    Two bands are needed because OpenCV's H-channel wraps at 180:
      Band 1: pure orange       (H ≈  5–25)
      Band 2: orange-red wrap   (H ≈ 160–180)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv,
                        np.array([hue_lo1, sat_lo, val_lo], np.uint8),
                        np.array([hue_hi1, 255,    255],    np.uint8))
    mask2 = cv2.inRange(hsv,
                        np.array([hue_lo2, sat_lo, val_lo], np.uint8),
                        np.array([hue_hi2, 255,    255],    np.uint8))
    return cv2.bitwise_or(mask1, mask2)


def clean_mask(mask: np.ndarray, close_px: int) -> np.ndarray:
    """
    Morphological closing (bridges dashes) followed by opening (noise removal).
    """
    diameter = 2 * close_px + 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest 8-connected foreground component.
    Discards isolated arrows, distance markers, and label blobs.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_labels <= 1:
        return mask

    # stats[:,CC_STAT_AREA] includes background at index 0; skip it
    biggest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == biggest_label).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Skeletonization
# ---------------------------------------------------------------------------

def skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Reduce the binary mask to a 1-pixel centreline.

    Primary method: scikit-image Zhang-Suen algorithm.
    Fallback: iterative morphological thinning (no dependency required).
    """
    try:
        from skimage.morphology import skeletonize as _sk
        skel = _sk(mask > 0)
        return skel.astype(np.uint8) * 255
    except ImportError:
        pass

    # Iterative erosion fallback (primitive; produces slightly thicker lines)
    skel   = np.zeros_like(mask)
    temp   = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded  = cv2.erode(temp, kernel)
        dilated = cv2.dilate(eroded, kernel)
        sub     = cv2.subtract(temp, dilated)
        skel    = cv2.bitwise_or(skel, sub)
        temp    = eroded
        if cv2.countNonZero(temp) == 0:
            break
    return skel


# ---------------------------------------------------------------------------
# Path ordering
# ---------------------------------------------------------------------------

def _build_adjacency(pts_set: set) -> dict:
    """8-connected adjacency dict.  O(8 N) with O(1) set membership."""
    adj: dict = {}
    for (x, y) in pts_set:
        adj[(x, y)] = [
            (x + dx, y + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts_set
        ]
    return adj


def order_skeleton(skel: np.ndarray) -> list[tuple[int, int]]:
    """
    Order skeleton pixels into a single traversal path.

    Strategy
    --------
    1. Build 8-connected adjacency (O(N)).
    2. Find degree-1 endpoint pixels; start from one.
    3. Greedy walk: among unvisited neighbours pick the one with the
       fewest own unvisited neighbours (prefers path continuation over
       branching into a dead-end).
    4. At dead-ends: find the nearest unvisited pixel via KDTree and
       jump to it (handles small gaps in the skeleton).
    """
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return []

    all_pts  = set(zip(xs.tolist(), ys.tolist()))
    adj      = _build_adjacency(all_pts)
    unvisited: set = set(all_pts)

    endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    start     = endpoints[0] if endpoints else next(iter(all_pts))

    ordered: list[tuple[int, int]] = [start]
    unvisited.discard(start)

    # KDTree for efficient gap-bridging nearest-neighbour queries
    try:
        from scipy.spatial import KDTree as _KDTree
        _use_kdtree = True
    except ImportError:
        _use_kdtree = False

    all_arr = np.array(sorted(all_pts), dtype=np.int32)  # (N, 2) as [x, y]

    while unvisited:
        cur         = ordered[-1]
        unvis_nbrs  = [n for n in adj[cur] if n in unvisited]

        if unvis_nbrs:
            # Prefer the neighbour that keeps us on a straight path
            nxt = min(unvis_nbrs,
                      key=lambda n: sum(1 for m in adj[n] if m in unvisited))
            ordered.append(nxt)
            unvisited.discard(nxt)
        else:
            # Dead-end: jump to the nearest unvisited pixel
            uv_list = list(unvisited)
            uv_arr  = np.array(uv_list, dtype=np.int32)

            if _use_kdtree:
                from scipy.spatial import KDTree
                dist, idx = KDTree(uv_arr).query(np.array(cur, dtype=np.int32))
                nearest = tuple(uv_arr[idx].tolist())
            else:
                cx, cy  = cur
                nearest = min(uv_list,
                              key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

            ordered.append(nearest)
            unvisited.discard(nearest)

    return ordered


def subsample(path: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Uniformly subsample an ordered path to at most *target* points."""
    n = len(path)
    if n <= target:
        return path
    idx = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
    return [path[i] for i in idx]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(
    image: np.ndarray,
    path: list[tuple[int, int]],
) -> np.ndarray:
    vis = image.copy()
    if not path:
        return vis
    pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255),
                  thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(vis, path[0],  9, (0, 220, 0),   -1)   # green = start
    cv2.circle(vis, path[-1], 9, (255, 50, 50),  -1)   # blue  = end
    return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment the orange marathon path and extract an ordered skeleton."
    )
    p.add_argument("--image",     required=True, help="Marathon schematic image path")
    p.add_argument("--hue-lo1",  type=int, default=5,   help="Primary orange lower hue  (default: 5)")
    p.add_argument("--hue-hi1",  type=int, default=25,  help="Primary orange upper hue  (default: 25)")
    p.add_argument("--hue-lo2",  type=int, default=160, help="Red-orange wrap lower hue (default: 160)")
    p.add_argument("--hue-hi2",  type=int, default=180, help="Red-orange wrap upper hue (default: 180)")
    p.add_argument("--sat-lo",   type=int, default=100, help="Min saturation            (default: 100)")
    p.add_argument("--val-lo",   type=int, default=80,  help="Min brightness            (default: 80)")
    p.add_argument("--close-px", type=int, default=15,
                   help="Closing radius in px to bridge dashes (default: 15)")
    p.add_argument("--subsample", type=int, default=500,
                   help="Target waypoint count after ordering (default: 500)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/07.path_extract/",
                   help="Output directory (default: ./path-to-gpx/output/07.path_extract/)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    print(f"[INFO] loading image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print("[ERROR] cv2.imread failed — check the file path.")
        return 1
    h, w = image.shape[:2]
    print(f"[INFO] image size: {w}×{h}")

    # ---- Segmentation --------------------------------------------------------
    print("[INFO] segmenting orange pixels …")
    raw_mask = segment_orange(
        image,
        args.hue_lo1, args.hue_hi1,
        args.hue_lo2, args.hue_hi2,
        args.sat_lo,  args.val_lo,
    )
    raw_px = int(np.count_nonzero(raw_mask))
    print(f"  raw orange pixels: {raw_px}")

    if raw_px == 0:
        print(
            "[ERROR] no orange pixels detected.\n"
            "  Adjust --hue-lo1/--hue-hi1 (default 5-25) and --sat-lo/--val-lo.\n"
            "  Tip: inspect the image in an HSV colour picker to find the exact hue."
        )
        return 1

    cleaned = clean_mask(raw_mask, args.close_px)
    print(f"  after closing+opening: {int(np.count_nonzero(cleaned))} px")

    main_blob = largest_component(cleaned)
    print(f"  largest component:     {int(np.count_nonzero(main_blob))} px")

    mask_path_out = output_dir / f"{stem}_orange_mask.png"
    cv2.imwrite(str(mask_path_out), main_blob)
    print(f"  mask → {mask_path_out.name}")

    # ---- Skeletonization -----------------------------------------------------
    print("[INFO] skeletonizing (Zhang-Suen) …")
    skel = skeletonize(main_blob)
    skel_px = int(np.count_nonzero(skel))
    print(f"  skeleton pixels: {skel_px}")

    skel_img_path = output_dir / f"{stem}_skeleton.png"
    cv2.imwrite(str(skel_img_path), skel)
    print(f"  skeleton → {skel_img_path.name}")

    if skel_px < 10:
        print(
            "[ERROR] skeleton is nearly empty after skeletonization.\n"
            "  The segmentation mask may be too noisy or fragmented.\n"
            "  Try increasing --close-px."
        )
        return 1

    # ---- Path ordering -------------------------------------------------------
    print("[INFO] ordering skeleton pixels into a traversal path …")
    ordered = order_skeleton(skel)
    print(f"  ordered pixels: {len(ordered)}")

    waypoints = subsample(ordered, args.subsample)
    print(f"  subsampled to:  {len(waypoints)} waypoints")

    json_out = output_dir / f"{stem}_skeleton_path.json"
    json_out.write_text(json.dumps({
        "image":               str(image_path),
        "image_width":         w,
        "image_height":        h,
        "total_skeleton_px":   skel_px,
        "ordered_px":          len(ordered),
        "waypoint_count":      len(waypoints),
        "segmentation_params": {
            "hue_lo1": args.hue_lo1, "hue_hi1": args.hue_hi1,
            "hue_lo2": args.hue_lo2, "hue_hi2": args.hue_hi2,
            "sat_lo":  args.sat_lo,  "val_lo":  args.val_lo,
            "close_px": args.close_px,
        },
        "waypoints": [{"px": int(x), "py": int(y)} for x, y in waypoints],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  path JSON → {json_out.name}")

    # ---- Visualisation -------------------------------------------------------
    vis = draw_overlay(image, waypoints)
    vis_path = output_dir / f"{stem}_path_visualized.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"  visualisation → {vis_path.name}")

    print(f"\n[DONE] {len(waypoints)} waypoints → {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Marathon path mask post-processing pipeline.

Steps:
    1. Score-based noise filtering  (area × circularity × extent × skeleton_length)
    2. Distance-transform based iterative fragment connection
    3. Residual fragment removal
    4. Skeletonization + spur pruning
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize as _skeletonize
from skimage.morphology import thin as _thin


def postprocess_mask(
    mask: np.ndarray,
    area_thresh: int = 250,
    circ_thresh: float = 0.5,
    skel_thresh: int = 400,
    max_distance: float = 150.0,
    min_fragment_size: int = 0,
    line_thickness: int = 2,
    morph_close_size: int = 10,
    final_size_thresh: int = 0,
    spur_length: int = 20,
    skel_morph_close: int = 0,
) -> np.ndarray:
    """
    Run the full post-processing pipeline on a binary mask.

    Args:
        mask: (H, W) uint8, path=255 background=0

    Returns:
        Skeletonized result as (H, W) uint8
    """
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = _filter_noise(mask, area_thresh, circ_thresh, skel_thresh)
    mask = _connect_fragments(mask, max_distance, min_fragment_size, line_thickness, morph_close_size)
    mask = _remove_small_fragments(mask, final_size_thresh)
    return _skeletonize_mask(mask, spur_length, skel_morph_close)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _components(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


# ── Step 1: Score-based noise filtering ──────────────────────────────────────

def _circularity(area: int, comp_mask: np.ndarray) -> float:
    """circularity = 4πA / P²  (원형=1, 길쭉한 형태→0)"""
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return float(4.0 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _filter_noise(
    mask: np.ndarray,
    area_thresh: int,
    circ_thresh: float,
    skel_thresh: int,
) -> np.ndarray:
    """
    Score-based noise filtering.

    Scoring per component (main path is always preserved):
        area < area_thresh                                   → +2
        circularity > circ_thresh                            → +1
        extent > 0.45  AND  not path-like (aspect < 2.5)    → +1
        skel_len < skel_thresh AND extent > 0.45 AND not path-like → +1

    Components with score >= 2 are removed.
    """
    num, labels, stats, _ = _components(mask)
    if num <= 1:
        return mask

    main_label = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])
    out = mask.copy()

    for lbl in range(1, num):
        if lbl == main_label:
            continue

        area   = int(stats[lbl, cv2.CC_STAT_AREA])
        width  = int(stats[lbl, cv2.CC_STAT_WIDTH])
        height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        bbox_area       = max(width * height, 1)
        extent          = float(area / bbox_area)
        short_side      = max(min(width, height), 1)
        bbox_aspect     = float(max(width, height) / short_side)
        path_like       = bbox_aspect >= 2.5

        comp_mask = (labels == lbl).astype(np.uint8) * 255
        skel_len  = int(_skeletonize(comp_mask > 0).sum())

        score = 0
        if area < area_thresh:
            score += 2
        if _circularity(area, comp_mask) > circ_thresh:
            score += 1
        if extent > 0.45 and not path_like:
            score += 1
        if skel_len < skel_thresh and extent > 0.45 and not path_like:
            score += 1

        if score >= 2:
            out[labels == lbl] = 0

    return out


# ── Step 2: Distance-transform based fragment connection ──────────────────────

def _extract_boundary_points(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.empty((0, 2), dtype=np.int64)
    kernel   = np.ones((3, 3), dtype=np.uint8)
    boundary = binary & (cv2.erode(binary, kernel, iterations=1) == 0)
    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([rows, cols])


def _build_main_dt(main_mask: np.ndarray):
    """Build distance transform of main mask's background for fragment snapping."""
    binary = (main_mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return None, None, np.empty((0, 2), dtype=np.int64)
    src = (binary == 0).astype(np.uint8)
    main_dt, main_dt_labels = cv2.distanceTransformWithLabels(
        src, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    main_dt_coords = np.column_stack(np.where(binary > 0))
    return main_dt, main_dt_labels, main_dt_coords


def _find_dt_connection(main_dt, main_dt_labels, main_dt_coords, frag_mask):
    """Find closest (frag_pt, main_pt, distance) pair via distance transform."""
    frag_pts = _extract_boundary_points(frag_mask)
    if len(frag_pts) == 0:
        rows, cols = np.where(frag_mask > 0)
        if len(rows) == 0:
            return None, None, float("inf")
        frag_pts = np.column_stack([rows, cols])

    dists    = main_dt[frag_pts[:, 0], frag_pts[:, 1]]
    best_idx = int(np.argmin(dists))
    frag_pt  = frag_pts[best_idx]

    label    = int(main_dt_labels[frag_pt[0], frag_pt[1]])
    main_idx = label - 1
    if main_idx < 0 or main_idx >= len(main_dt_coords):
        return None, None, float("inf")

    main_pt = main_dt_coords[main_idx]
    return frag_pt, main_pt, float(dists[best_idx])


def _connect_fragments(
    mask: np.ndarray,
    max_distance: float,
    min_fragment_size: int,
    line_thickness: int,
    morph_close_size: int,
) -> np.ndarray:
    """
    Iteratively merge fragments into the main path using distance-transform
    proximity. Each iteration draws a midpoint polyline bridge to connect the
    nearest fragment to the main path.
    """
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    for _ in range(10_000):
        num, labels, stats, _ = _components(mask)
        if num <= 2:
            break

        by_area  = sorted(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA], reverse=True)
        main_lbl = by_area[0]
        frags    = [l for l in by_area[1:] if stats[l, cv2.CC_STAT_AREA] >= min_fragment_size]
        if not frags:
            break

        main_mask_u8 = (labels == main_lbl).astype(np.uint8) * 255
        main_dt, main_dt_labels, main_dt_coords = _build_main_dt(main_mask_u8)
        if main_dt is None or len(main_dt_coords) == 0:
            break

        best_dist, best_main_pt, best_frag_pt, best_fmask = float("inf"), None, None, None
        for frag in frags:
            fmask = (labels == frag).astype(np.uint8) * 255
            frag_pt, main_pt, dist = _find_dt_connection(main_dt, main_dt_labels, main_dt_coords, fmask)
            if main_pt is not None and dist < best_dist:
                best_dist, best_main_pt, best_frag_pt, best_fmask = dist, main_pt, frag_pt, fmask

        if best_dist > max_distance:
            break

        mask[best_fmask > 0] = 255
        # midpoint polyline bridge: main_pt → midpoint → frag_pt
        mid_pt = np.rint(
            (best_main_pt.astype(np.float32) + best_frag_pt.astype(np.float32)) / 2.0
        ).astype(np.int32)
        pts = np.array([
            [int(best_main_pt[1]), int(best_main_pt[0])],
            [int(mid_pt[1]),       int(mid_pt[0])],
            [int(best_frag_pt[1]), int(best_frag_pt[0])],
        ], dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)

    return mask


# ── Step 3: Residual fragment removal ────────────────────────────────────────

def _remove_small_fragments(mask: np.ndarray, min_size: int) -> np.ndarray:
    num, labels, stats, _ = _components(mask)
    if num <= 1:
        return mask
    main = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])
    out  = np.zeros_like(mask)
    for lbl in range(1, num):
        if lbl == main or (min_size > 0 and stats[lbl, cv2.CC_STAT_AREA] >= min_size):
            out[labels == lbl] = 255
    return out


# ── Step 4: Skeletonization + spur pruning ───────────────────────────────────

def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    u8     = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(u8, ddepth=-1, kernel=kernel)


def _prune_spurs(skel: np.ndarray, spur_length: int) -> np.ndarray:
    """
    Iteratively remove spur segments (branches that touch exactly one branch
    point and are shorter than spur_length pixels).

    n_bp == 1 rule:
      0  → isolated path, never remove
      1  → one end is an endpoint → spur, remove
      2  → bridge between two branch points, never remove
    """
    pruned   = skel.copy()
    dilate_k = np.ones((3, 3), dtype=np.uint8)

    for _ in range(500):
        nc          = _neighbor_count(pruned)
        branch_pts  = pruned & (nc >= 3)
        segments_only = pruned & ~branch_pts

        num_labels, labels = cv2.connectedComponents(
            segments_only.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            break

        removed = False
        for lbl in range(1, num_labels):
            seg = labels == lbl
            if int(seg.sum()) >= spur_length:
                continue
            dilated = cv2.dilate(seg.astype(np.uint8), dilate_k) > 0
            n_bp    = int(np.sum(branch_pts & dilated & ~seg))
            if n_bp == 1:
                pruned[seg] = False
                removed = True

        if not removed:
            break

    return pruned


def _skeletonize_mask(mask: np.ndarray, spur_length: int, morph_close_size: int) -> np.ndarray:
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    skel = _skeletonize(mask > 127)
    skel = _thin(skel, max_num_iter=1)
    if spur_length > 0:
        skel = _prune_spurs(skel, spur_length)
    return skel.astype(np.uint8) * 255

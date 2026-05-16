"""
Marathon path mask post-processing pipeline.

Steps:
    1. Shape-based noise filtering   (area × circularity × skeleton_length)
    2. Iterative fragment connection  (endpoint-based)
    3. Residual fragment removal
    4. Skeletonization + spur pruning
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize as _skeletonize


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
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    skel = _skeletonize(mask > 127)
    if spur_length > 0:
        skel = _prune_spurs(skel, spur_length)
    return skel.astype(np.uint8) * 255


# ── Internal helpers ──────────────────────────────────────────────────────────

def _components(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _filter_noise(mask, area_thresh, circ_thresh, skel_thresh):
    num, labels, stats, _ = _components(mask)
    if num <= 1:
        return mask

    main = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])
    out = mask.copy()

    for lbl in range(1, num):
        if lbl == main:
            continue
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        comp = (labels == lbl).astype(np.uint8) * 255
        if area < area_thresh and _circularity(area, comp) > circ_thresh and int(_skeletonize(comp > 0).sum()) < skel_thresh:
            out[labels == lbl] = 0

    return out


def _circularity(area: int, comp_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return float(4.0 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def _connect_fragments(mask, max_distance, min_fragment_size, line_thickness, morph_close_size):
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    for _ in range(10_000):
        num, labels, stats, _ = _components(mask)
        if num <= 2:
            break

        by_area = sorted(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA], reverse=True)
        main_lbl = by_area[0]
        frags = [l for l in by_area[1:] if stats[l, cv2.CC_STAT_AREA] >= min_fragment_size]
        if not frags:
            break

        main_pts = _endpoints((labels == main_lbl).astype(np.uint8) * 255)
        if len(main_pts) == 0:
            break

        best_dist, best_mp, best_fp, best_fmask = float("inf"), None, None, None
        for frag in frags:
            fmask = (labels == frag).astype(np.uint8) * 255
            mp, fp, dist = _closest_pair(main_pts, _endpoints(fmask))
            if mp is not None and dist < best_dist:
                best_dist, best_mp, best_fp, best_fmask = dist, mp, fp, fmask

        if best_dist > max_distance:
            break

        mask[best_fmask > 0] = 255
        cv2.line(mask, (int(best_mp[1]), int(best_mp[0])), (int(best_fp[1]), int(best_fp[0])), 255, line_thickness)

    return mask


def _endpoints(mask: np.ndarray) -> np.ndarray:
    skel = _skeletonize(mask > 127)
    nbr = cv2.filter2D(skel.astype(np.uint8), -1, np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8))
    rows, cols = np.where(skel & (nbr == 1))
    if len(rows) > 0:
        return np.column_stack([rows, cols])
    rows, cols = np.where(skel)
    if len(rows) > 0:
        return np.column_stack([rows, cols])
    rows, cols = np.where(mask > 0)
    return np.column_stack([rows, cols]) if len(rows) > 0 else np.empty((0, 2), dtype=np.int64)


def _closest_pair(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return None, None, float("inf")
    dists = np.sqrt(((a[:, np.newaxis] - b[np.newaxis]) ** 2).sum(2))
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return a[idx[0]], b[idx[1]], float(dists[idx])


def _remove_small_fragments(mask, min_size):
    num, labels, stats, _ = _components(mask)
    if num <= 1:
        return mask
    main = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    for lbl in range(1, num):
        if lbl == main or (min_size > 0 and stats[lbl, cv2.CC_STAT_AREA] >= min_size):
            out[labels == lbl] = 255
    return out


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(u8, ddepth=-1, kernel=kernel)


def _prune_spurs(skel: np.ndarray, spur_length: int) -> np.ndarray:
    pruned = skel.copy()
    dilate_k = np.ones((3, 3), dtype=np.uint8)

    for _ in range(500):
        nc = _neighbor_count(pruned)
        branch_pts = pruned & (nc >= 3)
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
            n_bp = int(np.sum(branch_pts & dilated & ~seg))

            if n_bp == 1:
                pruned[seg] = False
                removed = True

        if not removed:
            break

    return pruned

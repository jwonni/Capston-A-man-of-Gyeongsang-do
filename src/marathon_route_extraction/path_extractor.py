"""
Skeletonization and ordered path extraction.

Entry point:
    extract_ordered_path(mask_arr, start_xy, end_xy)
        → list of (x, y) pixel coordinates ordered start → end
        → None if no path can be found
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from skimage.morphology import skeletonize as _skeletonize

# 그래프 단순화
from .graph_simplifier import (
    Graph,
    simplify_graph,
    find_nearest_node,
    restore_detailed_path,
    extract_keypoints,
)

_OFFSETS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


# ── Zhang-Suen thinning ───────────────────────────────────────────────────────

def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide skeleton of a binary bool/uint8 array."""
    img = binary.astype(np.uint8).copy()

    def transitions(nbrs: tuple) -> int:
        return sum(nbrs[i] == 0 and nbrs[i + 1] == 1 for i in range(len(nbrs) - 1))

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove: list[tuple[int, int]] = []
            padded = np.pad(img, 1, mode="constant")
            for y, x in np.argwhere(img > 0):
                py, px = int(y) + 1, int(x) + 1
                p2 = int(padded[py - 1, px])
                p3 = int(padded[py - 1, px + 1])
                p4 = int(padded[py,     px + 1])
                p5 = int(padded[py + 1, px + 1])
                p6 = int(padded[py + 1, px])
                p7 = int(padded[py + 1, px - 1])
                p8 = int(padded[py,     px - 1])
                p9 = int(padded[py - 1, px - 1])

                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if b < 2 or b > 6:
                    continue
                a = transitions((p2, p3, p4, p5, p6, p7, p8, p9, p2))
                if a != 1:
                    continue
                if step == 0:
                    if p2 * p4 * p6 != 0:
                        continue
                    if p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0:
                        continue
                    if p2 * p6 * p8 != 0:
                        continue
                to_remove.append((int(y), int(x)))

            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0

    return img.astype(bool)


# ── Graph + path helpers ──────────────────────────────────────────────────────

def _skeleton_to_graph(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Build adjacency list keyed by (row, col) = (y, x)."""
    points = [tuple(int(v) for v in p) for p in np.argwhere(skeleton)]
    point_set = set(points)
    graph: dict[tuple, list] = {p: [] for p in points}
    for y, x in points:
        for dy, dx in _OFFSETS_8:
            nb = (y + dy, x + dx)
            if nb in point_set:
                graph[(y, x)].append(nb)
    return graph


def _find_nearest_skeleton_point(
    skeleton: np.ndarray,
    cx: int,
    cy: int,
) -> Optional[tuple[int, int]]:
    """Return the skeleton (row, col) closest to canvas click (cx=col, cy=row)."""
    pts = np.argwhere(skeleton)
    if len(pts) == 0:
        return None
    dists = (pts[:, 1] - cx) ** 2 + (pts[:, 0] - cy) ** 2
    idx = int(np.argmin(dists))
    return (int(pts[idx, 0]), int(pts[idx, 1]))


def _bfs_path(
    graph: dict[tuple, list],
    start: tuple[int, int],
    end: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """BFS shortest path from start to end; returns list of (row, col) or None."""
    if start == end:
        return [start]
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple, Optional[tuple]] = {start: None}
    while queue:
        node = queue.popleft()
        for nb in graph.get(node, []):
            if nb not in parent:
                parent[nb] = node
                if nb == end:
                    path: list[tuple[int, int]] = []
                    cur: Optional[tuple] = end
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                queue.append(nb)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_ordered_path(
    mask_arr: np.ndarray,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    tau: float = 3.0,
    angle_thresh: float = 20.0,
    min_dist: float = 8.0,
) -> Optional[list[tuple[int, int]]]:
    """
    Skeletonize mask_arr and return an ordered pixel list from start to end.

    Args:
        mask_arr:  (H, W) uint8 array — foreground pixels have value 255
        start_xy:  (x, y) user-clicked start point in mask coordinates
        end_xy:    (x, y) user-clicked end point in mask coordinates

    Returns:
        List of (x, y) tuples ordered start → end, or None if no path is found.
    """
    binary   = mask_arr > 127
    # postprocess_mask already returns a skimage skeleton; re-running the slow
    # pure-Python zhang_suen_thinning on it breaks junctions and takes minutes.
    # Use the same skimage routine for speed and consistency.
    skeleton = _skeletonize(binary)
    graph = simplify_graph(skeleton, tau)  # 그래프 단순화

    start_yx = find_nearest_node(graph, cy=start_xy[1], cx=start_xy[0])
    end_yx   = find_nearest_node(graph, cy=end_xy[1],   cx=end_xy[0])

    if start_yx is None or end_yx is None:
        return None

    path_yx = _bfs_path(graph, start_yx, end_yx)
    if path_yx is None:
        return None
    detailed_yx  = restore_detailed_path(graph, path_yx)
    keypoints_yx = extract_keypoints(detailed_yx, graph, angle_thresh, min_dist)

    # Convert (row, col) → (x, y)
    return [(x, y) for y, x in keypoints_yx]
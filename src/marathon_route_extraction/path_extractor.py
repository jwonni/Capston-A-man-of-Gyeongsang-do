"""
Skeletonization and ordered path extraction.

Entry point:
    extract_ordered_path(mask_arr, start_xy, end_xy)
        → list of (x, y) pixel coordinates ordered start → end
        → None if no path can be found
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from skimage.morphology import skeletonize as _skeletonize

from src.config import RDP_EPSILON

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


def _perp_dist(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> float:
    """Perpendicular distance from point p to line segment a-b."""
    if a == b:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    num = abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
    den = math.hypot(b[0] - a[0], b[1] - a[1])
    return num / den


def _rdp(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    """Ramer-Douglas-Peucker line simplification."""
    if len(points) < 3:
        return list(points)

    start, end = points[0], points[-1]
    max_dist, max_idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], start, end)
        if d > max_dist:
            max_dist, max_idx = d, i

    if max_dist > epsilon:
        left  = _rdp(points[:max_idx + 1], epsilon)
        right = _rdp(points[max_idx:], epsilon)
        return left[:-1] + right

    return [start, end]


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
    epsilon: float = RDP_EPSILON,
) -> Optional[list[tuple[int, int]]]:
    """
    Skeletonize mask_arr and return an ordered pixel list from start to end.

    Args:
        mask_arr:  (H, W) uint8 array — foreground pixels have value 255
        start_xy:  (x, y) user-clicked start point in mask coordinates
        end_xy:    (x, y) user-clicked end point in mask coordinates
        epsilon:   RDP simplification threshold in pixels

    Returns:
        List of (x, y) tuples ordered start → end, or None if no path is found.
    """
    binary   = mask_arr > 127
    skeleton = _skeletonize(binary)
    graph    = _skeleton_to_graph(skeleton)

    start_yx = _find_nearest_skeleton_point(skeleton, cx=start_xy[0], cy=start_xy[1])
    end_yx   = _find_nearest_skeleton_point(skeleton, cx=end_xy[0],   cy=end_xy[1])

    if start_yx is None or end_yx is None:
        return None

    path_yx = _bfs_path(graph, start_yx, end_yx)
    if path_yx is None:
        return None

    simplified_yx = _rdp(path_yx, epsilon)

    # Convert (row, col) → (x, y)
    return [(x, y) for y, x in simplified_yx]


def auto_extract_ordered_path(mask_arr: np.ndarray, epsilon: float = RDP_EPSILON) -> Optional[dict]:
    """
    Automatically extract an ordered path without manual start/end selection.

    Direction heuristic (reproducible):
      vertical span ≥ horizontal span  → bottom-to-top
        start = bottommost degree-1 endpoint, end = topmost
      otherwise                         → left-to-right
        start = leftmost  degree-1 endpoint, end = rightmost

    Falls back to extreme skeleton pixels when no degree-1 endpoints exist.

    Returns:
        {"start": [x, y], "end": [x, y], "path": [[x, y], ...]}
        or None if no path can be found.
    """
    binary   = mask_arr > 127
    skeleton = _skeletonize(binary)
    graph    = _skeleton_to_graph(skeleton)

    pts = np.argwhere(skeleton)          # each row: [row=y, col=x]
    if len(pts) == 0:
        return None

    endpoints = [n for n in graph if len(graph[n]) == 1]

    y_vals = pts[:, 0]
    x_vals = pts[:, 1]
    v_span = int(y_vals.max()) - int(y_vals.min())
    h_span = int(x_vals.max()) - int(x_vals.min())

    if v_span >= h_span:
        # bottom-to-top: start = max-y endpoint, end = min-y endpoint
        if len(endpoints) >= 2:
            start_yx = max(endpoints, key=lambda p: p[0])
            end_yx   = min(endpoints, key=lambda p: p[0])
        else:
            start_yx = tuple(int(v) for v in pts[int(np.argmax(y_vals))])
            end_yx   = tuple(int(v) for v in pts[int(np.argmin(y_vals))])
    else:
        # left-to-right: start = min-x endpoint, end = max-x endpoint
        if len(endpoints) >= 2:
            start_yx = min(endpoints, key=lambda p: p[1])
            end_yx   = max(endpoints, key=lambda p: p[1])
        else:
            start_yx = tuple(int(v) for v in pts[int(np.argmin(x_vals))])
            end_yx   = tuple(int(v) for v in pts[int(np.argmax(x_vals))])

    path_yx = _bfs_path(graph, start_yx, end_yx)
    if path_yx is None:
        return None

    simplified_yx = _rdp(path_yx, epsilon)
    path_xy = [(int(x), int(y)) for y, x in simplified_yx]
    return {
        "start": [path_xy[0][0], path_xy[0][1]],
        "end":   [path_xy[-1][0], path_xy[-1][1]],
        "path":  path_xy,
    }
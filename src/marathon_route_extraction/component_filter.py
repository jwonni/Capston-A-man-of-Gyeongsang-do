from __future__ import annotations

from collections import deque

import numpy as np

NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def connected_components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components for a boolean mask."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy, dx in NEIGHBOR_OFFSETS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        components.append(component)

    return components


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only components whose area is at least min_area."""
    if min_area <= 0:
        return binary.copy()

    cleaned = np.zeros_like(binary, dtype=bool)
    for component in connected_components(binary):
        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[np.array(ys), np.array(xs)] = True

    return cleaned
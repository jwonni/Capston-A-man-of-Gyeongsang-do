"""
끊어진 마라톤 경로 연결 스크립트.

이 스크립트는 이진 마스크에서 경로가 끊어진 구간을 찾아 연결하고,
결과를 이미지로 저장한다.

핵심 흐름:
1) 마스크 로드 및 이진화
2) 작은 잡음 제거
3) Zhang-Suen skeletonization
4) skeleton endpoint 추출
5) endpoint 사이의 후보 연결
6) A* 기반 경로 연결
7) 연결 결과와 중간 결과를 시각화하여 저장

출력은 정성적 검사용 이미지와 최종 연결 마스크를 포함한다.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("src", None)


NEIGHBOR_OFFSETS_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def build_arg_parser() -> argparse.ArgumentParser:
    """커맨드라인 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="Connect broken marathon path masks and save visual previews."
    )
    parser.add_argument(
        "--input-mask",
        type=str,
        required=True,
        help="Path to a single binary mask image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/path_connection",
        help="Directory for connected masks and preview images.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Foreground threshold for grayscale masks.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20,
        help="Remove connected components smaller than this area.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=40,
        help="Maximum endpoint gap that will be considered for bridging.",
    )
    parser.add_argument(
        "--search-margin",
        type=int,
        default=24,
        help="Extra crop margin around a candidate gap during A* search.",
    )
    parser.add_argument(
        "--bridge-thickness",
        type=int,
        default=1,
        help="Thickness used when rasterizing the bridging path.",
    )
    return parser


def load_binary_mask(mask_path: Path, threshold: int) -> np.ndarray:
    """마스크 이미지를 불러와 foreground=True인 이진 배열로 변환한다."""
    mask = Image.open(mask_path).convert("L")
    mask_arr = np.asarray(mask, dtype=np.uint8)
    return mask_arr > threshold


def connected_components(binary: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-연결 성분을 구해 각 성분의 픽셀 좌표를 반환한다."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        queue: deque[Tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        component: List[Tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))

            for dy, dx in NEIGHBOR_OFFSETS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        components.append(component)

    return components


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """작은 잡음 성분을 제거하고, 남은 성분만 반환한다."""
    if min_area <= 0:
        return binary.copy()

    cleaned = np.zeros_like(binary, dtype=bool)
    components = connected_components(binary)
    for component in components:
        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[np.array(ys), np.array(xs)] = True
    return cleaned


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    """각 픽셀의 8-이웃 foreground 개수를 계산한다."""
    arr = binary.astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant")
    counts = np.zeros_like(arr, dtype=np.uint8)

    for dy, dx in NEIGHBOR_OFFSETS_8:
        counts += padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]

    return counts


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning으로 skeleton을 생성한다.

    이 구현은 외부 의존성 없이 동작하도록 순수 numpy 루프로 작성했다.
    입력은 이진 마스크이며, 출력은 1픽셀 두께에 가까운 skeleton이다.
    """
    img = binary.astype(np.uint8).copy()

    def transitions(neighbors: Sequence[int]) -> int:
        return sum((neighbors[i] == 0 and neighbors[i + 1] == 1) for i in range(len(neighbors) - 1))

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove: List[Tuple[int, int]] = []
            padded = np.pad(img, 1, mode="constant")
            foreground_points = np.argwhere(img > 0)

            for y, x in foreground_points:
                py, px = y + 1, x + 1
                p2 = int(padded[py - 1, px])
                p3 = int(padded[py - 1, px + 1])
                p4 = int(padded[py, px + 1])
                p5 = int(padded[py + 1, px + 1])
                p6 = int(padded[py + 1, px])
                p7 = int(padded[py + 1, px - 1])
                p8 = int(padded[py, px - 1])
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

                to_remove.append((y, x))

            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0

    return img.astype(bool)


def find_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """skeleton의 endpoint(8-이웃이 1개인 픽셀)를 찾는다."""
    counts = neighbor_count(skeleton)
    endpoints = np.argwhere((skeleton > 0) & (counts == 1))
    return [(int(y), int(x)) for y, x in endpoints]


def endpoint_direction(skeleton: np.ndarray, point: Tuple[int, int]) -> Optional[np.ndarray]:
    """endpoint에서 안쪽을 향하는 방향 벡터를 계산한다."""
    y, x = point
    height, width = skeleton.shape

    for dy, dx in NEIGHBOR_OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]:
            vec = np.array([nx - x, ny - y], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                return vec / norm
    return None


def crop_bounds(
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    shape: Tuple[int, int],
    margin: int,
) -> Tuple[int, int, int, int]:
    """두 점을 감싸는 crop 영역을 계산한다."""
    y1, x1 = point_a
    y2, x2 = point_b
    height, width = shape

    top = max(0, min(y1, y2) - margin)
    bottom = min(height - 1, max(y1, y2) + margin)
    left = max(0, min(x1, x2) - margin)
    right = min(width - 1, max(x1, x2) + margin)
    return top, bottom, left, right


def local_support_cost(binary_crop: np.ndarray, foreground_penalty: float = 4.0) -> np.ndarray:
    """주변 foreground 밀도를 이용해 탐색 비용 지도를 만든다.

    이미 경로에 가까운 픽셀은 낮은 비용, 배경에만 있는 픽셀은 높은 비용을 갖는다.
    """
    arr = binary_crop.astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant")
    support = np.zeros_like(arr, dtype=np.float32)

    for dy, dx in NEIGHBOR_OFFSETS_8:
        support += padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]

    support = support / 8.0
    cost = 1.0 + foreground_penalty * (1.0 - support)
    cost[arr > 0] *= 0.35
    return cost


def astar_path(
    cost_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """8-이웃 격자에서 A* 경로를 구한다.

    좌표는 (y, x) 순서로 다룬다.
    """
    height, width = cost_map.shape

    def heuristic(node: Tuple[int, int]) -> float:
        dy = abs(node[0] - goal[0])
        dx = abs(node[1] - goal[1])
        return float(math.hypot(dx, dy))

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heappush(open_heap, (heuristic(start), 0.0, start))
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed: set[Tuple[int, int]] = set()

    while open_heap:
        _, current_g, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        cy, cx = current
        for dy, dx in NEIGHBOR_OFFSETS_8:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue

            step_cost = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
            candidate_g = current_g + step_cost * float((cost_map[cy, cx] + cost_map[ny, nx]) * 0.5)

            if candidate_g < g_score.get((ny, nx), float("inf")):
                g_score[(ny, nx)] = candidate_g
                parent[(ny, nx)] = current
                f_score = candidate_g + heuristic((ny, nx))
                heappush(open_heap, (f_score, candidate_g, (ny, nx)))

    return None


def draw_line_with_thickness(
    canvas: np.ndarray,
    points: Sequence[Tuple[int, int]],
    thickness: int,
) -> np.ndarray:
    """점들의 경로를 canvas 위에 그린다."""
    result = canvas.copy()
    radius = max(0, thickness - 1)

    for y, x in points:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1]:
                        result[ny, nx] = True

    return result


def pair_endpoints(
    endpoints: Sequence[Tuple[int, int]],
    skeleton: np.ndarray,
    max_gap: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """끝점 후보를 거리와 방향으로 평가해 연결 쌍을 고른다."""
    candidates: List[Tuple[float, float, Tuple[int, int], Tuple[int, int]]] = []
    endpoint_dirs: Dict[Tuple[int, int], Optional[np.ndarray]] = {
        endpoint: endpoint_direction(skeleton, endpoint) for endpoint in endpoints
    }

    for index, point_a in enumerate(endpoints):
        for point_b in endpoints[index + 1 :]:
            dy = point_b[0] - point_a[0]
            dx = point_b[1] - point_a[1]
            distance = math.hypot(dx, dy)
            if distance > max_gap:
                continue

            dir_a = endpoint_dirs.get(point_a)
            dir_b = endpoint_dirs.get(point_b)
            if dir_a is None or dir_b is None:
                direction_score = 0.15
            else:
                vec_ab = np.array([dx, dy], dtype=np.float32)
                norm_ab = float(np.linalg.norm(vec_ab))
                if norm_ab == 0:
                    continue
                unit_ab = vec_ab / norm_ab
                unit_ba = -unit_ab
                score_a = max(0.0, float(np.dot(dir_a, unit_ab)))
                score_b = max(0.0, float(np.dot(dir_b, unit_ba)))
                direction_score = score_a * score_b

            combined_score = direction_score / (1.0 + distance)
            candidates.append((combined_score, distance, point_a, point_b))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    used: set[Tuple[int, int]] = set()
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for score, distance, point_a, point_b in candidates:
        if point_a in used or point_b in used:
            continue
        used.add(point_a)
        used.add(point_b)
        pairs.append((point_a, point_b))

    return pairs


def bresenham_line(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """두 점을 잇는 정수 격자 직선을 반환한다."""
    y0, x0 = start
    y1, x1 = goal
    points: List[Tuple[int, int]] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy

    return points


def dilate_binary(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 구조요소로 단순 dilation을 수행한다."""
    result = binary.copy()
    for _ in range(iterations):
        padded = np.pad(result.astype(np.uint8), 1, mode="constant")
        expanded = np.zeros_like(result, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= padded[1 + dy : 1 + dy + result.shape[0], 1 + dx : 1 + dx + result.shape[1]].astype(bool)
        result = expanded
    return result


def mask_to_rgb(binary: np.ndarray) -> Image.Image:
    """이진 마스크를 흰색 foreground / 검은색 background RGB 이미지로 변환한다."""
    gray = binary.astype(np.uint8) * 255
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def overlay_connections(
    base_mask: np.ndarray,
    skeleton: np.ndarray,
    connected_pairs: Sequence[Sequence[Tuple[int, int]]],
) -> Image.Image:
    """연결 결과를 시각화하기 위한 RGB overlay를 생성한다."""
    base = mask_to_rgb(base_mask)
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)

    skeleton_points = np.argwhere(skeleton)
    for y, x in skeleton_points:
        overlay.putpixel((int(x), int(y)), (60, 120, 255))

    for index, path in enumerate(connected_pairs):
        if not path:
            continue
        color = (255, max(40, 180 - index * 20), 40)
        for start, end in zip(path[:-1], path[1:]):
            draw.line((start[1], start[0], end[1], end[0]), fill=color, width=2)

        start_y, start_x = path[0]
        end_y, end_x = path[-1]
        draw.ellipse((start_x - 3, start_y - 3, start_x + 3, start_y + 3), outline=(0, 200, 0), width=2)
        draw.ellipse((end_x - 3, end_y - 3, end_x + 3, end_y + 3), outline=(255, 0, 0), width=2)

    return overlay


def add_title(image: Image.Image, title: str) -> Image.Image:
    """이미지 상단에 제목을 추가한다."""
    margin_top = 28
    canvas = Image.new("RGB", (image.width, image.height + margin_top), (255, 255, 255))
    canvas.paste(image, (0, margin_top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    return canvas


def make_contact_sheet(panels: Sequence[Tuple[str, Image.Image]]) -> Image.Image:
    """정성적 검사용 비교 이미지를 만든다."""
    titled = [add_title(image, title) for title, image in panels]
    max_width = max(image.width for image in titled)
    max_height = max(image.height for image in titled)

    cols = 2
    rows = int(math.ceil(len(titled) / cols))
    sheet = Image.new("RGB", (cols * max_width, rows * max_height), (245, 245, 245))

    for index, image in enumerate(titled):
        row = index // cols
        col = index % cols
        x = col * max_width
        y = row * max_height
        sheet.paste(image, (x, y))

    return sheet


def connect_broken_path(
    binary_mask: np.ndarray,
    max_gap: int,
    search_margin: int,
    bridge_thickness: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]:
    """끊어진 경로를 연결하고, skeleton과 연결된 마스크를 반환한다."""
    cleaned_mask = binary_mask.copy()
    skeleton = zhang_suen_thinning(cleaned_mask)

    endpoints = find_endpoints(skeleton)
    endpoint_pairs = pair_endpoints(endpoints, skeleton, max_gap=max_gap)

    connection_layer = np.zeros_like(cleaned_mask, dtype=bool)
    connected_pairs: List[List[Tuple[int, int]]] = []

    for point_a, point_b in endpoint_pairs:
        top, bottom, left, right = crop_bounds(point_a, point_b, cleaned_mask.shape, search_margin)
        binary_crop = cleaned_mask[top : bottom + 1, left : right + 1]
        cost_map = local_support_cost(binary_crop)

        start = (point_a[0] - top, point_a[1] - left)
        goal = (point_b[0] - top, point_b[1] - left)
        path = astar_path(cost_map, start, goal)

        if path is None:
            # A*가 실패하면 endpoint를 직선으로 잇는 보수적 fallback을 사용한다.
            path = bresenham_line(start, goal)

        global_path = [(y + top, x + left) for y, x in path]
        connected_pairs.append(global_path)
        for y, x in global_path:
            connection_layer[y, x] = True

    if bridge_thickness > 1:
        connection_layer = dilate_binary(connection_layer, iterations=bridge_thickness - 1)

    connected_mask = cleaned_mask | connection_layer
    connected_skeleton = zhang_suen_thinning(connected_mask)
    return connected_mask, connected_skeleton, connected_pairs


def save_outputs(
    input_mask_path: Path,
    output_dir: Path,
    connected_mask: np.ndarray,
    connected_skeleton: np.ndarray,
    connected_pairs: Sequence[Sequence[Tuple[int, int]]],
) -> None:
    """연결 결과와 미리보기 이미지를 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_mask_path.stem
    connected_mask_path = output_dir / f"{stem}_connected.png"
    skeleton_path = output_dir / f"{stem}_connected_skeleton.png"
    preview_path = output_dir / f"{stem}_connection_preview.png"

    connected_mask_image = mask_to_rgb(connected_mask)
    connected_skeleton_image = mask_to_rgb(connected_skeleton)
    overlay_image = overlay_connections(connected_mask, connected_skeleton, connected_pairs)

    connected_mask_image.save(connected_mask_path)
    connected_skeleton_image.save(skeleton_path)

    contact_sheet = make_contact_sheet(
        [
            ("Connected Mask", connected_mask_image),
            ("Connected Skeleton", connected_skeleton_image),
            ("Overlay Preview", overlay_image),
        ]
    )
    contact_sheet.save(preview_path)

    print(f"Connected mask saved to: {connected_mask_path}")
    print(f"Connected skeleton saved to: {skeleton_path}")
    print(f"Preview image saved to: {preview_path}")


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    input_mask_path = Path(args.input_mask)
    if not input_mask_path.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask_path}")

    output_dir = Path(args.output_dir)

    binary_mask = load_binary_mask(input_mask_path, threshold=args.threshold)
    cleaned_mask = remove_small_components(binary_mask, min_area=args.min_component_area)

    connected_mask, connected_skeleton, connected_pairs = connect_broken_path(
        cleaned_mask,
        max_gap=args.max_gap,
        search_margin=args.search_margin,
        bridge_thickness=args.bridge_thickness,
    )

    save_outputs(
        input_mask_path=input_mask_path,
        output_dir=output_dir,
        connected_mask=connected_mask,
        connected_skeleton=connected_skeleton,
        connected_pairs=connected_pairs,
    )


if __name__ == "__main__":
    main()
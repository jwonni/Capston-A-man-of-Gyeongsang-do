"""
마스킹 이미지에서 순서가 있는 경로 픽셀 리스트를 추출하는 스크립트.

이 스크립트는 다음을 수행한다.
1) 이진 마스크 로드 및 잡음 제거
2) Zhang-Suen skeletonization
3) skeleton 그래프에서 가장 긴 주 경로 추정
4) ordered pixel list를 (x, y) 형식으로 JSON 저장
5) 정성적 검사용 이미지 저장

중요한 점:
경로의 순서가 중요하므로, 단순한 foreground 좌표 수집이 아니라
graph traversal 기반의 ordered path extraction을 사용한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
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
        description="Extract ordered marathon path pixels from a binary mask."
    )
    parser.add_argument(
        "--input-mask",
        type=str,
        required=True,
        help="Path to the mask image to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/path_pixels",
        help="Directory for JSON and preview images.",
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
        "--json-name",
        type=str,
        default=None,
        help="Optional custom name for the output JSON file.",
    )
    return parser


def load_binary_mask(mask_path: Path, threshold: int) -> np.ndarray:
    """마스크 이미지를 foreground=True인 이진 배열로 읽는다."""
    # PIL로 마스크를 열고, grayscale로 변환한 후, numpy 배열로 바꾼다.
    mask = Image.open(mask_path).convert("L")
    mask_arr = np.asarray(mask, dtype=np.uint8)

    # threshold를 적용하여 foreground 픽셀을 True로 만든다.
    # 전체 이미지 배열에서 threshold보다 큰 픽셀을 foreground로 간주한다.
    return mask_arr > threshold


def connected_components(binary: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-연결 성분을 찾는다."""
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
    """작은 성분을 제거한다."""
    # 8-연결 성분(한 픽셀의 상하좌우 및 대각선 이웃을 모두 연결된 것으로 간주)을 찾는다.
    # 즉, 한 픽셀을 기준으로 8방향으로 쭉 연결된 픽셀들을 하나의 성분으로 묶는다. 이때, 대각선 방향도 연결로 간주한다.
    # 이를 통해 binary 배열에서 연결된 픽셀 그룹들을 찾아낸다. 각 그룹은 하나의 성분이 된다.
    # 각 성분의 픽셀 개수를 계산하여, min_area보다 작은 성분은 노이즈로 간주하고 제거한다. 결과는 원본과 같은 크기의 이진 배열이다.
    if min_area <= 0:
        return binary.copy()

    # cleaned 배열을 fasle로 초기화.
    cleaned = np.zeros_like(binary, dtype=bool)
    # connected_components 함수를 사용하여 binary 배열에서 8-연결 성분을 찾는다. 각 성분은 픽셀 좌표의 리스트로 표현된다.
    components = connected_components(binary)
    # 각 성분의 픽셀 개수를 계산하여, min_area보다 큰 성분만 cleaned 배열에서 True로 설정한다. 작은 성분은 False로 남겨둔다.
    for component in components:
        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[np.array(ys), np.array(xs)] = True
    return cleaned


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    """각 foreground 픽셀의 8-이웃 개수를 계산한다."""
    arr = binary.astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant")
    counts = np.zeros_like(arr, dtype=np.uint8)

    for dy, dx in NEIGHBOR_OFFSETS_8:
        counts += padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]

    return counts

# 핵심 알고리즘: Zhang-Suen thinning을 사용하여 skeleton을 만들고, BFS 기반의 그래프 탐색으로 가장 긴 주 경로를 추출한다.
# skeleton이란 원본 마스크의 중심선을 나타내는 얇은 선 형태의 표현이다. Zhang-Suen 알고리즘은 이진 이미지에서 skeleton을 추출하는 방법 중 하나로, 반복적으로 픽셀을 제거하여 중심선을 남긴다.
def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning으로 skeleton을 만든다."""
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


def skeleton_to_graph(skeleton: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """skeleton을 그래프로 변환한다.

    키는 (y, x) 좌표이며, 값은 8-이웃으로 연결된 좌표 리스트이다.
    """
    # 그래프 구조로 skeleton을 표현한다. skeleton에서 foreground 픽셀을 찾아서 각 픽셀을 그래프의 노드로 만들고, 8-이웃이 연결된 노드끼리 간선을 만든다. 결과는 (y, x) 좌표를 키로 하고, 연결된 이웃 좌표 리스트를 값으로 하는 딕셔너리 형태의 그래프이다.
    points = [tuple(point) for point in np.argwhere(skeleton)]
    point_set = set(points)
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {point: [] for point in points}

    for y, x in points:
        for dy, dx in NEIGHBOR_OFFSETS_8:
            neighbor = (y + dy, x + dx)
            if neighbor in point_set:
                graph[(y, x)].append(neighbor)

    return graph


def find_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """endpoint를 반환한다."""
    # skeleton에서 foreground 픽셀 중에서 8-이웃이 1개인 픽셀을 endpoint로 간주한다. neighbor_count 함수를 사용하여 각 픽셀의 이웃 개수를 계산하고, 이웃이 1개인 픽셀의 좌표를 리스트로 반환한다.
    counts = neighbor_count(skeleton)
    endpoints = np.argwhere((skeleton > 0) & (counts == 1))
    return [(int(y), int(x)) for y, x in endpoints]


def bfs_farthest(
    start: Tuple[int, int],
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Tuple[Tuple[int, int], int, Dict[Tuple[int, int], Tuple[int, int]]]:
    """BFS를 통해 시작점에서 가장 먼 점과 부모 맵을 찾는다."""
    queue: deque[Tuple[int, int]] = deque([start])
    distance: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                parent[neighbor] = node
                queue.append(neighbor)

    farthest = max(distance, key=lambda point: distance[point])
    return farthest, distance[farthest], parent


def reconstruct_path(
    parent: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """부모 맵을 따라 ordered path를 복원한다."""
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def deterministic_endpoint_order(
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """항상 같은 방향의 순서를 얻기 위한 결정적 정렬 기준을 적용한다."""
    if (point_a[0], point_a[1]) <= (point_b[0], point_b[1]):
        return point_a, point_b
    return point_b, point_a


def select_main_path(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """skeleton 그래프에서 가장 긴 주 경로를 선택한다."""
    graph = skeleton_to_graph(skeleton)
    points = list(graph.keys())
    if not points:
        return []

    endpoints = find_endpoints(skeleton)
    endpoint_set = set(endpoints)

    if len(endpoints) >= 2:
        best_start: Optional[Tuple[int, int]] = None
        best_end: Optional[Tuple[int, int]] = None
        best_distance = -1
        best_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for start in endpoints:
            farthest, distance, parent = bfs_farthest(start, graph)
            if farthest in endpoint_set and distance > best_distance:
                best_start = start
                best_end = farthest
                best_distance = distance
                best_parent = parent

        if best_start is not None and best_end is not None:
            path = reconstruct_path(best_parent, best_start, best_end)
            ordered_start, _ = deterministic_endpoint_order(best_start, best_end)
            if ordered_start != best_start:
                path = list(reversed(path))
            return path

    # endpoint가 없거나 충분하지 않으면, 그래프의 지름을 이용해 순서를 만든다.
    arbitrary_start = min(points, key=lambda point: (point[0], point[1]))
    first_end, _, _ = bfs_farthest(arbitrary_start, graph)
    second_end, _, parent = bfs_farthest(first_end, graph)
    path = reconstruct_path(parent, first_end, second_end)
    ordered_start, _ = deterministic_endpoint_order(first_end, second_end)
    if ordered_start != first_end:
        path = list(reversed(path))
    return path


def skeletonize_and_extract(binary_mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """skeleton과 ordered main path를 함께 추출한다."""
    skeleton = zhang_suen_thinning(binary_mask)
    ordered_path = select_main_path(skeleton)
    return skeleton, ordered_path


def mask_to_rgb(binary: np.ndarray) -> Image.Image:
    """이진 마스크를 RGB 이미지로 바꾼다."""
    gray = binary.astype(np.uint8) * 255
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def draw_ordered_path_overlay(
    base_mask: np.ndarray,
    skeleton: np.ndarray,
    ordered_path: Sequence[Tuple[int, int]],
) -> Image.Image:
    """순서가 있는 경로를 색상 그라디언트로 시각화한다."""
    overlay = mask_to_rgb(base_mask)
    draw = ImageDraw.Draw(overlay)

    skeleton_points = np.argwhere(skeleton)
    for y, x in skeleton_points:
        overlay.putpixel((int(x), int(y)), (80, 140, 255))

    if len(ordered_path) >= 2:
        total = len(ordered_path) - 1
        for index, (start, end) in enumerate(zip(ordered_path[:-1], ordered_path[1:])):
            ratio = index / max(1, total)
            color = (
                int(40 + 215 * ratio),
                int(120 + 100 * (1.0 - ratio)),
                int(255 * (1.0 - ratio)),
            )
            draw.line((start[0], start[1], end[0], end[1]), fill=color, width=2)

    if ordered_path:
        start_x, start_y = ordered_path[0]
        end_x, end_y = ordered_path[-1]
        draw.ellipse((start_x - 4, start_y - 4, start_x + 4, start_y + 4), outline=(0, 180, 0), width=2)
        draw.ellipse((end_x - 4, end_y - 4, end_x + 4, end_y + 4), outline=(255, 0, 0), width=2)
        draw.text((start_x + 6, start_y + 6), "start", fill=(0, 120, 0), font=ImageFont.load_default())
        draw.text((end_x + 6, end_y + 6), "end", fill=(180, 0, 0), font=ImageFont.load_default())

    return overlay


def add_title(image: Image.Image, title: str) -> Image.Image:
    """이미지 상단에 제목을 붙인다."""
    margin_top = 28
    canvas = Image.new("RGB", (image.width, image.height + margin_top), (255, 255, 255))
    canvas.paste(image, (0, margin_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
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


def save_outputs(
    input_mask_path: Path,
    output_dir: Path,
    binary_mask: np.ndarray,
    skeleton: np.ndarray,
    ordered_path_xy: Sequence[Tuple[int, int]],
    json_name: Optional[str],
) -> None:
    """JSON과 미리보기 이미지를 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_mask_path.stem
    resolved_json_name = json_name or f"{stem}_ordered_pixels.json"
    preview_name = f"{stem}_ordered_path_preview.png"

    ordered_path_list = [[int(x), int(y)] for x, y in ordered_path_xy]
    json_path = output_dir / resolved_json_name
    preview_path = output_dir / preview_name

    payload = {
        "input_mask": str(input_mask_path),
        "image_width": int(binary_mask.shape[1]),
        "image_height": int(binary_mask.shape[0]),
        "coordinate_format": "(x, y)",
        "ordered_pixels_xy": ordered_path_list,
        "num_pixels": len(ordered_path_list),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    mask_image = mask_to_rgb(binary_mask)
    skeleton_image = mask_to_rgb(skeleton)
    overlay_image = draw_ordered_path_overlay(binary_mask, skeleton, ordered_path_xy)

    contact_sheet = make_contact_sheet(
        [
            ("Original Mask", mask_image),
            ("Skeleton", skeleton_image),
            ("Ordered Path Overlay", overlay_image),
        ]
    )
    contact_sheet.save(preview_path)

    print(f"Ordered pixel list saved to: {json_path}")
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
    skeleton, ordered_path = skeletonize_and_extract(cleaned_mask)

    save_outputs(
        input_mask_path=input_mask_path,
        output_dir=output_dir,
        binary_mask=cleaned_mask,
        skeleton=skeleton,
        ordered_path_xy=ordered_path,
        json_name=args.json_name,
    )


if __name__ == "__main__":
    main()
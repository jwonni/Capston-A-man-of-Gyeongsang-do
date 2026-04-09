"""
작은 성분 제거(remove_small_components) 과정을 시각화하는 스크립트.

입력 마스크에 대해 아래 결과를 생성한다.
1) 이진화 결과
2) 제거 전 연결 성분 라벨 시각화
3) 작은 성분 제거 후 마스크
4) 제거된 픽셀만 강조한 오버레이
5) 간단한 통계 JSON
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    parser = argparse.ArgumentParser(description="Visualize remove_small_components process")
    parser.add_argument(
        "--input-mask",
        type=str,
        required=True,
        help="Path to input mask image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/component_filter_visualization",
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=130,
        help="Minimum area to keep components",
    )
    parser.add_argument(
        "--max-labeled-components",
        type=int,
        default=60,
        help="Maximum components to color individually in label preview",
    )
    return parser


def load_binary_mask(mask_path: Path, threshold: int) -> np.ndarray:
    """마스크를 읽어 이진 배열(bool)로 변환한다."""
    image = Image.open(mask_path).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    return arr > threshold


def connected_components(binary: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-연결 성분을 구한다."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack: List[Tuple[int, int]] = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component: List[Tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy, dx in NEIGHBOR_OFFSETS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        components.append(component)

    return components


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """픽셀 수가 min_area 미만인 성분을 제거한다."""
    if min_area <= 0:
        return binary.copy()

    cleaned = np.zeros_like(binary, dtype=bool)
    components = connected_components(binary)
    for component in components:
        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[np.array(ys), np.array(xs)] = True

    return cleaned


def bool_to_rgb(mask: np.ndarray) -> Image.Image:
    """bool 마스크를 흑백 RGB로 변환한다."""
    gray = mask.astype(np.uint8) * 255
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def component_label_preview(
    binary: np.ndarray,
    components: List[List[Tuple[int, int]]],
    max_components: int,
) -> Image.Image:
    """연결 성분을 색으로 구분해 보여준다."""
    canvas = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

    # 큰 성분을 먼저 표시하면 주요 성분 식별이 쉬워진다.
    sorted_components = sorted(components, key=len, reverse=True)
    palette = [
        (255, 90, 90),
        (90, 200, 255),
        (100, 230, 130),
        (255, 190, 70),
        (200, 120, 255),
        (255, 120, 190),
        (120, 255, 210),
        (210, 220, 120),
    ]

    for idx, component in enumerate(sorted_components[: max(1, max_components)]):
        color = palette[idx % len(palette)]
        ys, xs = zip(*component)
        canvas[np.array(ys), np.array(xs)] = np.array(color, dtype=np.uint8)

    # 색상 제한 밖의 성분은 회색으로 처리한다.
    for component in sorted_components[max(1, max_components) :]:
        ys, xs = zip(*component)
        canvas[np.array(ys), np.array(xs)] = np.array((150, 150, 150), dtype=np.uint8)

    return Image.fromarray(canvas)


def removed_overlay(original: np.ndarray, cleaned: np.ndarray) -> Image.Image:
    """제거된 픽셀을 빨간색으로 강조한다."""
    base = np.stack([original.astype(np.uint8) * 255] * 3, axis=-1)
    removed = original & (~cleaned)
    base[removed] = np.array([255, 40, 40], dtype=np.uint8)
    return Image.fromarray(base)


def with_title(image: Image.Image, title: str) -> Image.Image:
    """이미지 상단에 제목 텍스트를 붙인다."""
    top = 30
    canvas = Image.new("RGB", (image.width, image.height + top), (255, 255, 255))
    canvas.paste(image, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def contact_sheet(images: List[Image.Image], cols: int = 2) -> Image.Image:
    """여러 이미지를 그리드로 묶는다."""
    if not images:
        raise ValueError("No images to compose")

    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * max_w, rows * max_h), (245, 245, 245))

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        sheet.paste(img, (c * max_w, r * max_h))

    return sheet


def main() -> None:
    """CLI 진입점."""
    args = build_arg_parser().parse_args()

    input_mask_path = Path(args.input_mask)
    if not input_mask_path.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary = load_binary_mask(input_mask_path, threshold=args.threshold)
    components = connected_components(binary)
    cleaned = remove_small_components(binary, min_area=args.min_component_area)

    removed = binary & (~cleaned)
    kept_components = sum(1 for c in components if len(c) >= args.min_component_area)
    removed_components = len(components) - kept_components

    panel_original = with_title(
        bool_to_rgb(binary),
        f"Original Binary | pixels={int(np.sum(binary))}",
    )
    panel_labels = with_title(
        component_label_preview(binary, components, max_components=args.max_labeled_components),
        f"Connected Components | total={len(components)}",
    )
    panel_cleaned = with_title(
        bool_to_rgb(cleaned),
        f"After remove_small_components | pixels={int(np.sum(cleaned))}",
    )
    panel_removed = with_title(
        removed_overlay(binary, cleaned),
        f"Removed Pixels(RED) | pixels={int(np.sum(removed))}",
    )

    sheet = contact_sheet([panel_original, panel_labels, panel_cleaned, panel_removed], cols=2)

    stem = input_mask_path.stem
    sheet_path = output_dir / f"{stem}_component_filter_sheet.png"
    binary_path = output_dir / f"{stem}_binary.png"
    cleaned_path = output_dir / f"{stem}_cleaned_min_area_{args.min_component_area}.png"
    removed_path = output_dir / f"{stem}_removed_overlay.png"
    stats_path = output_dir / f"{stem}_component_filter_stats.json"

    sheet.save(sheet_path)
    bool_to_rgb(binary).save(binary_path)
    bool_to_rgb(cleaned).save(cleaned_path)
    removed_overlay(binary, cleaned).save(removed_path)

    stats = {
        "input_mask": str(input_mask_path),
        "threshold": int(args.threshold),
        "min_component_area": int(args.min_component_area),
        "total_components": int(len(components)),
        "kept_components": int(kept_components),
        "removed_components": int(removed_components),
        "original_foreground_pixels": int(np.sum(binary)),
        "cleaned_foreground_pixels": int(np.sum(cleaned)),
        "removed_pixels": int(np.sum(removed)),
        "outputs": {
            "sheet": str(sheet_path),
            "binary": str(binary_path),
            "cleaned": str(cleaned_path),
            "removed_overlay": str(removed_path),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Visualization sheet saved to: {sheet_path}")
    print(f"Stats JSON saved to: {stats_path}")


if __name__ == "__main__":
    main()

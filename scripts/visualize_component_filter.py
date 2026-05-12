"""
Visualize connected-component filtering on a binary mask.

Usage:
    python scripts/visualize_component_filter.py \
        --input-mask sample_route.png \
        --output-dir outputs/component_filter_visualization \
        --min-component-area 130
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.marathon_route_extraction.component_filter import connected_components, remove_small_components


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize remove_small_components process")
    parser.add_argument("--input-mask", type=str, required=True, help="Path to input mask image")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/component_filter_visualization",
        help="Output directory",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=130,
        help="Minimum area to keep components",
    )
    return parser


def load_binary_mask(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    return arr > 0


def bool_to_rgb(mask: np.ndarray) -> Image.Image:
    gray = mask.astype(np.uint8) * 255
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def component_label_preview(components: list[list[tuple[int, int]]], height: int, width: int) -> Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
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

    for idx, component in enumerate(sorted_components[:60]):
        color = palette[idx % len(palette)]
        ys, xs = zip(*component)
        canvas[np.array(ys), np.array(xs)] = np.array(color, dtype=np.uint8)

    for component in sorted_components[60:]:
        ys, xs = zip(*component)
        canvas[np.array(ys), np.array(xs)] = np.array((150, 150, 150), dtype=np.uint8)

    return Image.fromarray(canvas)


def removed_overlay(original: np.ndarray, cleaned: np.ndarray) -> Image.Image:
    base = np.stack([original.astype(np.uint8) * 255] * 3, axis=-1)
    removed = original & (~cleaned)
    base[removed] = np.array([255, 40, 40], dtype=np.uint8)
    return Image.fromarray(base)


def with_title(image: Image.Image, title: str) -> Image.Image:
    top = 30
    canvas = Image.new("RGB", (image.width, image.height + top), (255, 255, 255))
    canvas.paste(image, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def contact_sheet(images: list[Image.Image], cols: int = 2) -> Image.Image:
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
    args = build_arg_parser().parse_args()

    input_mask_path = Path(args.input_mask)
    if not input_mask_path.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary = load_binary_mask(input_mask_path)
    components = connected_components(binary)
    cleaned = remove_small_components(binary, min_area=args.min_component_area)

    removed = binary & (~cleaned)
    kept_components = sum(1 for c in components if len(c) >= args.min_component_area)
    removed_components = len(components) - kept_components

    panel_original = with_title(bool_to_rgb(binary), f"Original Binary | pixels={int(np.sum(binary))}")
    panel_labels = with_title(
        component_label_preview(components, binary.shape[0], binary.shape[1]),
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

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class OcrItem:
    idx: int
    text: str
    matched_place: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class TileBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)


def _load_ocr_items(ocr_json_path: Path) -> tuple[str, list[OcrItem]]:
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    image_path = str(data.get("image", ""))
    items: list[OcrItem] = []

    detections = data.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("invalid OCR JSON: 'detections' must be a list")

    for d in detections:
        if not isinstance(d, dict):
            continue
        try:
            items.append(
                OcrItem(
                    idx=int(d.get("idx", 0)),
                    text=str(d.get("text", "")),
                    matched_place=str(d.get("matched_place", "")),
                    confidence=float(d.get("confidence", 0.0)),
                    center_x=float(d.get("center_x", 0.0)),
                    center_y=float(d.get("center_y", 0.0)),
                    width=float(d.get("width", 0.0)),
                    height=float(d.get("height", 0.0)),
                )
            )
        except Exception:
            continue

    return image_path, items


def _clamp_box(cx: float, cy: float, size: int, image_w: int, image_h: int) -> TileBox:
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    if x0 < 0:
        x1 += -x0
        x0 = 0
    if y0 < 0:
        y1 += -y0
        y0 = 0
    if x1 > image_w:
        x0 -= x1 - image_w
        x1 = image_w
    if y1 > image_h:
        y0 -= y1 - image_h
        y1 = image_h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image_w, x1)
    y1 = min(image_h, y1)

    return TileBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _keyword_match_score(item: OcrItem, primary_keyword: str) -> float:
    if not primary_keyword:
        return 0.0

    needle = primary_keyword.strip().lower()
    if not needle:
        return 0.0

    haystack = f"{item.text} {item.matched_place}".lower()
    return 1.0 if needle in haystack else 0.0


def _importance_score(item: OcrItem, image_area: float, primary_keyword: str) -> float:
    box_area = max(1.0, item.width * item.height)
    relative_area = box_area / max(1.0, image_area)
    # Confidence + box size + optional keyword hit bonus.
    return item.confidence * 10.0 + relative_area * 2.0 + _keyword_match_score(item, primary_keyword) * 100.0


def _pick_primary_item(items: list[OcrItem], min_confidence: float, image_area: float, primary_keyword: str) -> tuple[OcrItem, float]:
    filtered = [d for d in items if d.confidence >= min_confidence]
    if not filtered:
        raise RuntimeError("no OCR detection passed min-confidence; lower --min-confidence")

    scored = [
        (
            _importance_score(d, image_area=image_area, primary_keyword=primary_keyword),
            d,
        )
        for d in filtered
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], float(scored[0][0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a single primary OCR-guided tile from marathon image."
        )
    )
    parser.add_argument("--image", required=False, help="Input image path. If omitted, use OCR JSON image path")
    parser.add_argument("--ocr-json", required=True, help="Path to OCR detections JSON from 01_ocr.py")
    parser.add_argument(
        "--output-dir",
        default="./path-to-gpx/output/ocr_tiles/",
        help="Output directory for tile images and metadata",
    )
    parser.add_argument("--tile-size", type=int, default=700, help="Square tile size in pixels")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum OCR confidence")
    parser.add_argument(
        "--primary-keyword",
        default="",
        help="Optional preferred keyword. If found, it is strongly prioritized.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json)
    if not ocr_json_path.exists() or not ocr_json_path.is_file():
        raise FileNotFoundError(f"ocr json not found: {ocr_json_path}")

    json_image_path, ocr_items = _load_ocr_items(ocr_json_path)

    if args.image:
        image_path = Path(args.image)
    elif json_image_path:
        image_path = Path(json_image_path)
    else:
        raise ValueError("image path is missing. Provide --image or include 'image' in OCR JSON")

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to load image: {image_path}")

    image_h, image_w = image.shape[:2]
    tile_size = int(max(64, args.tile_size))
    min_confidence = float(max(0.0, min(1.0, args.min_confidence)))
    image_area = float(image_h * image_w)
    primary_item, primary_score = _pick_primary_item(
        items=ocr_items,
        min_confidence=min_confidence,
        image_area=image_area,
        primary_keyword=args.primary_keyword,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_tile = _clamp_box(primary_item.center_x, primary_item.center_y, tile_size, image_w, image_h)

    stem = image_path.stem
    vis = image.copy()

    tile_name = f"{stem}_ocrtile_primary.png"
    tile_path = output_dir / tile_name
    crop = image[primary_tile.y0 : primary_tile.y1, primary_tile.x0 : primary_tile.x1]
    cv2.imwrite(str(tile_path), crop)

    cv2.rectangle(vis, (primary_tile.x0, primary_tile.y0), (primary_tile.x1, primary_tile.y1), (0, 255, 0), 2)
    label = f"primary idx={primary_item.idx} text={primary_item.text[:20]}"
    cv2.putText(
        vis,
        label,
        (primary_tile.x0 + 4, max(16, primary_tile.y0 + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    vis_path = output_dir / f"{stem}_ocr_tiles_visualized.png"
    cv2.imwrite(str(vis_path), vis)

    report = {
        "image": str(image_path),
        "ocr_json": str(ocr_json_path),
        "tile_size": tile_size,
        "min_confidence": min_confidence,
        "primary_keyword": args.primary_keyword,
        "primary_score": primary_score,
        "input_ocr_count": len(ocr_items),
        "selected_ocr_count": len([d for d in ocr_items if d.confidence >= min_confidence]),
        "generated_tile_count": 1,
        "primary_detection": {
            "idx": primary_item.idx,
            "text": primary_item.text,
            "matched_place": primary_item.matched_place,
            "confidence": primary_item.confidence,
            "center_x": primary_item.center_x,
            "center_y": primary_item.center_y,
            "width": primary_item.width,
            "height": primary_item.height,
        },
        "tile": {
            "x0": primary_tile.x0,
            "y0": primary_tile.y0,
            "x1": primary_tile.x1,
            "y1": primary_tile.y1,
            "width": primary_tile.w,
            "height": primary_tile.h,
        },
        "outputs": {
            "visualization": str(vis_path),
            "tile_images": [str(tile_path)],
        },
    }

    report_path = output_dir / f"{stem}_ocr_tiles_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] selected primary keyword text: {primary_item.text}")
    print(f"[DONE] generated tiles: 1")
    print(f"[DONE] primary tile: {tile_path}")
    print(f"[DONE] visualization: {vis_path}")
    print(f"[DONE] report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""05_anchor_patches.py — Semantic Anchor Patch Extraction

For each geocoded landmark:
  1. Crop a ``patch_size x patch_size`` "source patch" from the marathon
     schematic image (Image A) centred on the OCR detection bounding box.
  2. Download a matching "reference tile" from OSM at a high zoom level
     (default z17, ~1 m/px) centred on the geocoded (lat, lon).

The produced patch pairs are the inputs to 06_loftr_match.py for
cross-domain feature matching.

Inputs
------
--image          Path to the marathon schematic image (Image A)
--ocr-json       OCR detections JSON produced by 01_ocr.py
--geocode-json   Geocoding candidates JSON produced by 02_geo_coding.py
--zoom           OSM zoom level for reference tiles (default: 17, range 14-19)
--patch-size     Side length of both patches in pixels (default: 512)
--output-dir     Directory to write outputs

Outputs
-------
<output-dir>/<stem>_anchors.json
<output-dir>/patches/<stem>_anchor_<idx>_src.png   — source patch from Image A
<output-dir>/patches/<stem>_anchor_<idx>_ref.png   — OSM reference tile
<output-dir>/tile_cache/                            — raw 256px OSM tile cache
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


OSM_TILE_PX = 256
_UA = "MarathonGPXPipeline/1.0"


# ---------------------------------------------------------------------------
# Web-Mercator helpers
# ---------------------------------------------------------------------------

def _clamp_lat(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def latlon_to_tile_float(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional tile position (tx, ty) for a lat/lon at the given zoom."""
    lat = _clamp_lat(lat)
    lon = ((lon + 180.0) % 360.0) - 180.0
    n = 1 << zoom
    tx = (lon + 180.0) / 360.0 * n
    ty = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return tx, ty


def tile_float_to_latlon(tx: float, ty: float, zoom: int) -> tuple[float, float]:
    """Convert fractional tile position back to (lat, lon)."""
    n = 1 << zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lat, lon


# ---------------------------------------------------------------------------
# OSM tile fetching
# ---------------------------------------------------------------------------

def _fetch_one_tile(tx: int, ty: int, zoom: int, cache_dir: Path) -> np.ndarray | None:
    """Download and cache a single 256×256 OSM raster tile."""
    tile_path = cache_dir / f"z{zoom}_{tx}_{ty}.png"
    if not tile_path.exists():
        url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                tile_path.write_bytes(resp.read())
            time.sleep(0.15)  # polite rate-limiting per OSM tile usage policy
        except Exception as exc:
            print(f"    [WARN] tile fetch {zoom}/{tx}/{ty} failed: {exc}")
            return None
    img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    return img


def fetch_reference_tile(
    lat: float,
    lon: float,
    zoom: int,
    patch_size: int,
    cache_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Build a ``patch_size x patch_size`` composite OSM image centred on
    (lat, lon), and return it alongside a geotransform dict.

    The geotransform lets callers convert any pixel (px, py) in the returned
    image to (lat, lon):

        tx_f = top_left_tx_float + px / OSM_TILE_PX
        ty_f = top_left_ty_float + py / OSM_TILE_PX
        lat, lon = tile_float_to_latlon(tx_f, ty_f, zoom)
    """
    tx_f, ty_f = latlon_to_tile_float(lat, lon, zoom)
    tx0, ty0 = int(tx_f), int(ty_f)

    # Sub-tile pixel offset of the requested centre
    cx_in_tile = (tx_f - tx0) * OSM_TILE_PX
    cy_in_tile = (ty_f - ty0) * OSM_TILE_PX

    half = patch_size // 2
    # Enough tile rows/cols to cover the patch no matter the sub-tile offset
    t_radius = math.ceil(patch_size / OSM_TILE_PX) + 1

    canvas_tiles = 2 * t_radius + 1
    canvas_px = canvas_tiles * OSM_TILE_PX
    canvas = np.zeros((canvas_px, canvas_px, 3), dtype=np.uint8)

    for dy in range(-t_radius, t_radius + 1):
        for dx in range(-t_radius, t_radius + 1):
            tile = _fetch_one_tile(tx0 + dx, ty0 + dy, zoom, cache_dir)
            if tile is not None:
                px_off = (dx + t_radius) * OSM_TILE_PX
                py_off = (dy + t_radius) * OSM_TILE_PX
                th, tw = tile.shape[:2]
                canvas[py_off:py_off + th, px_off:px_off + tw] = tile

    # Position of (lat, lon) in the canvas
    cx_canvas = t_radius * OSM_TILE_PX + cx_in_tile
    cy_canvas = t_radius * OSM_TILE_PX + cy_in_tile

    x0 = int(round(cx_canvas - half))
    y0 = int(round(cy_canvas - half))
    x1, y1 = x0 + patch_size, y0 + patch_size

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(canvas_px, x1), min(canvas_px, y1)

    crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    crop[y0c - y0:y0c - y0 + (y1c - y0c),
         x0c - x0:x0c - x0 + (x1c - x0c)] = canvas[y0c:y1c, x0c:x1c]

    geotransform = {
        "zoom": zoom,
        "center_lat": lat,
        "center_lon": lon,
        "center_pixel_x": half,
        "center_pixel_y": half,
        "top_left_tx_float": tx_f - half / OSM_TILE_PX,
        "top_left_ty_float": ty_f - half / OSM_TILE_PX,
        "osm_tile_px": OSM_TILE_PX,
        "patch_size": patch_size,
    }
    return crop, geotransform


# ---------------------------------------------------------------------------
# Source patch extraction
# ---------------------------------------------------------------------------

def crop_source_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size: int,
) -> tuple[np.ndarray, dict]:
    """
    Crop a ``patch_size x patch_size`` region from *image* centred on
    (center_x, center_y), padding with zeros if near the image border.

    Returns (patch, offset_info) where offset_info encodes the affine
    mapping:  patch_pixel (px, py) → image_pixel (px + offset_x, py + offset_y).
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x1, y1 = x0 + patch_size, y0 + patch_size

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
    patch[y0c - y0:y0c - y0 + (y1c - y0c),
          x0c - x0:x0c - x0 + (x1c - x0c)] = image[y0c:y1c, x0c:x1c]

    return patch, {
        "offset_x": x0,
        "offset_y": y0,
        "center_in_image_x": float(center_x),
        "center_in_image_y": float(center_y),
        "center_in_patch_x": half,
        "center_in_patch_y": half,
        "patch_size": patch_size,
        "image_width": w,
        "image_height": h,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crop semantic anchor patches from Image A and fetch matching OSM reference tiles."
    )
    p.add_argument("--image",        required=True, help="Marathon schematic image (Image A)")
    p.add_argument("--ocr-json",     required=True, help="OCR detections JSON from 01_ocr.py")
    p.add_argument("--geocode-json", required=True, help="Geocode candidates JSON from 02_geo_coding.py")
    p.add_argument("--zoom",       type=int,   default=17,  help="OSM zoom level (default: 17)")
    p.add_argument("--patch-size", type=int,   default=512, help="Patch size in pixels (default: 512)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/anchors/",
                   help="Output directory (default: ./path-to-gpx/output/anchors/)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    image_path      = Path(args.image)
    ocr_json_path   = Path(args.ocr_json)
    geo_json_path   = Path(args.geocode_json)

    for path, name in [
        (image_path, "--image"),
        (ocr_json_path, "--ocr-json"),
        (geo_json_path, "--geocode-json"),
    ]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}")
            return 1

    output_dir   = Path(args.output_dir)
    patch_dir    = output_dir / "patches"
    cache_dir    = output_dir / "tile_cache"
    for d in (output_dir, patch_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERROR] failed to read image: {image_path}")
        return 1
    h_img, w_img = image.shape[:2]
    print(f"[INFO] image size: {w_img}×{h_img}")

    ocr_data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    geo_data = json.loads(geo_json_path.read_text(encoding="utf-8"))

    # Build OCR pixel-coordinate lookup keyed by detection idx
    ocr_by_idx: dict[int, dict] = {
        int(d["idx"]): d for d in ocr_data.get("detections", [])
    }

    # Keep only geocoded candidates with valid coordinates
    geocoded = [
        c for c in geo_data.get("geocode_candidates", [])
        if c.get("geocoded") and c.get("lat") is not None and c.get("lon") is not None
    ]

    if not geocoded:
        print(
            "[ERROR] no geocoded candidates in geocode JSON.\n"
            "        Run 02_geo_coding.py and ensure at least one place name resolves."
        )
        return 1

    stem = image_path.stem
    anchors: list[dict] = []

    for cand in geocoded:
        idx  = int(cand["idx"])
        lat  = float(cand["lat"])
        lon  = float(cand["lon"])
        text = cand.get("text", "")

        ocr_det = ocr_by_idx.get(idx)
        if ocr_det is None:
            print(f"  [WARN] OCR detection not found for idx={idx} ('{text}'), skipping.")
            continue

        cx = float(ocr_det["center_x"])
        cy = float(ocr_det["center_y"])
        print(f"\n[INFO] anchor idx={idx}  '{text}'  img=({cx:.0f},{cy:.0f})  lat={lat:.6f} lon={lon:.6f}")

        # Source patch
        src_patch, src_offset = crop_source_patch(image, cx, cy, args.patch_size)
        src_path = patch_dir / f"{stem}_anchor_{idx}_src.png"
        cv2.imwrite(str(src_path), src_patch)

        # Reference tile
        print(f"  fetching OSM reference tile at zoom={args.zoom} …")
        ref_tile, geotransform = fetch_reference_tile(
            lat, lon, args.zoom, args.patch_size, cache_dir
        )
        ref_path = patch_dir / f"{stem}_anchor_{idx}_ref.png"
        cv2.imwrite(str(ref_path), ref_tile)

        anchors.append({
            "idx":            idx,
            "text":           text,
            "lat":            lat,
            "lon":            lon,
            "display_name":   cand.get("display_name", ""),
            "image_center_x": cx,
            "image_center_y": cy,
            "src_patch":      str(src_path),
            "src_offset":     src_offset,
            "ref_tile":       str(ref_path),
            "geotransform":   geotransform,
        })
        print(f"  → src={src_path.name}  ref={ref_path.name}")

    if not anchors:
        print("[ERROR] no anchor pairs produced. Check that OCR and geocoding outputs are non-empty.")
        return 1

    out_json = output_dir / f"{stem}_anchors.json"
    out_json.write_text(json.dumps({
        "image":        str(image_path),
        "image_width":  w_img,
        "image_height": h_img,
        "zoom":         args.zoom,
        "patch_size":   args.patch_size,
        "anchor_count": len(anchors),
        "anchors":      anchors,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] {len(anchors)} anchor pair(s) → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""07_tps_align.py — Hierarchical Alignment + Thin-Plate Spline Model

Builds a Thin-Plate Spline (TPS) that maps any pixel (px, py) in the
marathon schematic image (Image A) to real-world (lat, lon).

Two control-point sources are merged in priority order:

  Phase 1 – Coarse (geocoded anchors):
    Each OCR-detected landmark in Image A carries a (center_x, center_y)
    and a Nominatim-geocoded (lat, lon).  These are direct correspondences
    that do not require any feature matching.  Nominatim returns the centroid
    of an administrative region, so accuracy is ~100–500 m — sufficient for
    coarse alignment of a 20–42 km course.

  Phase 2 – Fine (LoFTR-derived points):
    06_loftr_match.py maps inlier patch correspondences back to Image-A
    pixels and converts them to (lat, lon) via the OSM tile geotransform.
    These are sub-metre accurate at zoom ≥ 17, so they are preferred and
    deduplicated first.

The final TPS is fitted with scipy.interpolate.RBFInterpolator (kernel=
'thin_plate_spline', scipy ≥ 1.7).  For ≤ 3 control points an affine
least-squares fit is used instead.

Inputs
------
--anchors-json   Output of 05_anchor_patches.py   (geocoded control points)
--loftr-json     Output of 06_loftr_match.py      (optional; fine points)
--smoothing      RBF smoothing parameter (default: 1e-4)
--min-dist-px    Min pixel distance for duplicate removal (default: 10.0)
--output-dir     Directory to write output (default: ./path-to-gpx/output/tps/)

Output
------
<output-dir>/<stem>_tps_model.json   — control points + quality metrics
                                       (TPS is re-fitted at runtime from CPs)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Control-point collection
# ---------------------------------------------------------------------------

def _geocoded_cps(anchors_data: dict) -> list[dict]:
    """Extract direct geocoded control points from 05_anchor_patches.py output."""
    cps = []
    for a in anchors_data.get("anchors", []):
        cps.append({
            "img_x":  a["image_center_x"],
            "img_y":  a["image_center_y"],
            "lat":    a["lat"],
            "lon":    a["lon"],
            "source": "geocoded",
            "idx":    a["idx"],
            "text":   a.get("text", ""),
        })
    return cps


def _loftr_cps(loftr_data: dict) -> list[dict]:
    """Extract LoFTR-derived control points from 06_loftr_match.py output."""
    cps = []
    for res in loftr_data.get("anchor_results", []):
        if res.get("status") != "ok":
            continue
        for cp in res.get("control_points", []):
            cps.append({
                "img_x":  cp["img_x"],
                "img_y":  cp["img_y"],
                "lat":    cp["lat"],
                "lon":    cp["lon"],
                "source": "loftr",
                "idx":    res["idx"],
            })
    return cps


def deduplicate(cps: list[dict], min_dist_px: float) -> list[dict]:
    """
    Remove near-duplicate control points.
    LoFTR points take priority over geocoded ones (they are more precise).
    """
    if not cps:
        return []

    priority = {"loftr": 0, "geocoded": 1}
    ordered  = sorted(cps, key=lambda c: (priority.get(c["source"], 9), c.get("idx", 0)))

    kept: list[dict] = []
    for cp in ordered:
        px, py = cp["img_x"], cp["img_y"]
        if any(math.hypot(px - k["img_x"], py - k["img_y"]) < min_dist_px for k in kept):
            continue
        kept.append(cp)
    return kept


# ---------------------------------------------------------------------------
# TPS / Affine fitting
# ---------------------------------------------------------------------------

def _build_tps(src: np.ndarray, dst: np.ndarray, smoothing: float):
    """Fit a Thin-Plate Spline (scipy ≥ 1.7 required).

    src: (N, 2) — [img_x, img_y]
    dst: (N, 2) — [lat, lon]
    Returns an RBFInterpolator instance.
    """
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)


def _build_affine(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares affine fit for < 4 control points.
    src: (N, 2), dst: (N, 2)
    Returns a callable f(query: (M,2)) → (M,2).
    """
    # Augment: [x, y, 1] @ A ≈ [lat, lon]
    A_src = np.column_stack([src, np.ones(len(src))])  # (N, 3)
    A, _, _, _ = np.linalg.lstsq(A_src, dst, rcond=None)  # (3, 2)

    def _predict(query: np.ndarray) -> np.ndarray:
        q_aug = np.column_stack([query, np.ones(len(query))])
        return q_aug @ A

    return _predict


def build_model(cps: list[dict], smoothing: float):
    """Build the appropriate spatial model given the number of control points."""
    src = np.array([[c["img_x"], c["img_y"]] for c in cps], dtype=np.float64)
    dst = np.array([[c["lat"],   c["lon"]]   for c in cps], dtype=np.float64)

    if len(cps) >= 4:
        try:
            tps = _build_tps(src, dst, smoothing)
            return tps, "tps", src, dst
        except Exception as exc:
            print(f"  [WARN] TPS failed ({exc}), falling back to affine.")

    affine_fn = _build_affine(src, dst)
    return affine_fn, "affine", src, dst


def self_residuals(model, src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """Compute self-residuals (training-set fit quality) in degrees."""
    try:
        pred = model(src)
        res = np.linalg.norm(pred - dst, axis=1)
        return float(np.mean(res)), float(np.max(res))
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------

def _coverage(cps: list[dict], image_hw: tuple[int, int]) -> dict:
    if len(cps) < 2:
        return {"n_cps": len(cps), "quality": "insufficient", "coverage_pct": 0.0}

    h, w = image_hw
    xs = [c["img_x"] / w for c in cps]
    ys = [c["img_y"] / h for c in cps]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    cov = x_span * y_span * 100.0

    n = len(cps)
    quality = "poor"
    if n >= 4  and cov > 5.0:  quality = "fair"
    if n >= 6  and cov > 15.0: quality = "good"
    if n >= 10 and cov > 30.0: quality = "excellent"

    return {
        "n_cps":        n,
        "quality":      quality,
        "coverage_pct": round(cov, 2),
        "x_span_pct":   round(x_span * 100.0, 2),
        "y_span_pct":   round(y_span * 100.0, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build TPS / affine alignment model from geocoded + LoFTR control points."
    )
    p.add_argument("--anchors-json", required=True,
                   help="Anchors JSON from 05_anchor_patches.py")
    p.add_argument("--loftr-json", default=None,
                   help="LoFTR matches JSON from 06_loftr_match.py (optional)")
    p.add_argument("--smoothing", type=float, default=1e-4,
                   help="TPS smoothing parameter (default: 1e-4; increase for noisier inputs)")
    p.add_argument("--min-dist-px", type=float, default=10.0,
                   help="Pixel-distance threshold for duplicate removal (default: 10.0)")
    p.add_argument("--output-dir", default="./path-to-gpx/output/tps/",
                   help="Output directory (default: ./path-to-gpx/output/tps/)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    anchors_path = Path(args.anchors_json)
    if not anchors_path.exists():
        print(f"[ERROR] anchors JSON not found: {anchors_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors_data = json.loads(anchors_path.read_text(encoding="utf-8"))
    stem         = Path(anchors_data["image"]).stem
    image_hw     = (anchors_data["image_height"], anchors_data["image_width"])

    # Phase 1 — geocoded control points
    geo_cps = _geocoded_cps(anchors_data)
    print(f"[INFO] geocoded control points: {len(geo_cps)}")

    # Phase 2 — LoFTR-derived control points (optional)
    loftr_cps: list[dict] = []
    if args.loftr_json:
        loftr_path = Path(args.loftr_json)
        if loftr_path.exists():
            loftr_data = json.loads(loftr_path.read_text(encoding="utf-8"))
            loftr_cps  = _loftr_cps(loftr_data)
            print(f"[INFO] LoFTR control points (pre-dedup): {len(loftr_cps)}")
        else:
            print(f"[WARN] LoFTR JSON not found: {loftr_path}  → fine phase skipped.")

    all_cps = deduplicate(geo_cps + loftr_cps, min_dist_px=args.min_dist_px)
    print(f"[INFO] control points after deduplication: {len(all_cps)}")
    print(f"       sources — geocoded={sum(1 for c in all_cps if c['source']=='geocoded')} "
          f"loftr={sum(1 for c in all_cps if c['source']=='loftr')}")

    cov = _coverage(all_cps, image_hw)
    print(f"[INFO] spatial coverage: {cov}")

    if len(all_cps) < 2:
        print(
            "[ERROR] need ≥ 2 control points. "
            "Ensure OCR detects at least 2 geocodable landmarks, "
            "or run 06_loftr_match.py to add LoFTR-derived points."
        )
        return 1

    if len(all_cps) < 4:
        print(f"[WARN] only {len(all_cps)} control points — using affine transform (TPS needs ≥ 4).")

    # Build model and compute self-residuals
    model, model_type, src_arr, dst_arr = build_model(all_cps, args.smoothing)
    mean_res, max_res = self_residuals(model, src_arr, dst_arr)
    print(f"[INFO] model_type={model_type}  self-residual mean={mean_res:.6f}°  max={max_res:.6f}°")
    if mean_res > 0.05:
        print(
            "  [WARN] high self-residual — the control points may be inconsistent. "
            "Check that OCR pixel positions and geocoded coordinates match the same landmarks."
        )

    out_json = output_dir / f"{stem}_tps_model.json"
    out_json.write_text(json.dumps({
        "image":        anchors_data["image"],
        "image_width":  anchors_data["image_width"],
        "image_height": anchors_data["image_height"],
        "model_type":   model_type,
        "smoothing":    args.smoothing,
        "n_geocoded_cps": sum(1 for c in all_cps if c["source"] == "geocoded"),
        "n_loftr_cps":    sum(1 for c in all_cps if c["source"] == "loftr"),
        "n_total_cps":    len(all_cps),
        "coverage_quality":            cov,
        "tps_self_residual_mean_deg":  mean_res,
        "tps_self_residual_max_deg":   max_res,
        "control_points": all_cps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] {model_type.upper()} model ({len(all_cps)} CPs) → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

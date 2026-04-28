"""run_full_pipeline.py — End-to-End Marathon GPX Pipeline Orchestrator

Runs all pipeline stages in sequence:

  Stage 1  — 01_ocr.py           OCR text detection
  Stage 2  — 02_geo_coding.py    Geocode OCR place names → lat/lon
  Stage 3  — 03_osm.py           Download global OSM tile for context
  Stage 5  — 05_anchor_patches.py  Crop semantic anchor patches + OSM ref tiles
  Stage 6  — 06_loftr_match.py   LoFTR + MAGSAC++ cross-domain matching
  Stage 7  — 07_tps_align.py     Hierarchical TPS control-point model
  Stage 8  — 08_path_extract.py  Orange path segmentation + skeletonization
  Stage 9  — 09_osrm_gpx.py      OSRM snapping + GPX generation

Usage
-----
    python run_full_pipeline.py --image <path/to/marathon.jpg>

Key options
-----------
--output-dir     Root output directory (default: ./path-to-gpx/output/)
--zoom-anchor    OSM zoom for anchor ref tiles (default: 17)
--zoom-global    OSM zoom for global overview tile (default: 14)
--patch-size     Anchor patch size in pixels (default: 512)
--subsample      Target skeleton waypoint count (default: 500)
--osrm-url       OSRM server (default: http://router.project-osrm.org)
--profile        OSRM profile: foot | bike | car (default: foot)
--skip-loftr     Skip stage 6 (LoFTR); use geocoded CPs only for TPS
--skip-osrm      Skip stage 9 OSRM snapping; write raw TPS GPX
--hue-lo1/hi1    Orange hue band 1 (default: 5-25)
--hue-lo2/hi2    Orange hue band 2, red-orange wrap (default: 160-180)
--sat-lo         Min orange saturation (default: 100)
--val-lo         Min orange value/brightness (default: 80)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {label}")
    print(f"{'='*60}")
    print("  cmd:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage '{label}' exited with code {result.returncode}.\n"
            "Fix the error above and re-run, or use --skip-* flags to bypass optional stages."
        )


def _script(name: str) -> str:
    return str(SRC_DIR / name)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end marathon schematic → GPX pipeline."
    )
    p.add_argument("--image", required=True,
                   help="Input marathon schematic image (JPEG/PNG)")

    # Output layout
    p.add_argument("--output-dir", default="./path-to-gpx/output/",
                   help="Root output directory (default: ./path-to-gpx/output/)")

    # OCR / geocoding
    p.add_argument("--lang",           default="korean",
                   choices=["korean", "english"], help="OCR language (default: korean)")
    p.add_argument("--min-confidence", type=float, default=0.2,
                   help="Min OCR confidence (default: 0.2)")
    p.add_argument("--geocode-top-k",  type=int,   default=5,
                   help="Top-K candidates to geocode (default: 5)")
    p.add_argument("--primary-keyword", default="",
                   help="Preferred landmark keyword for OSM tile selection")

    # Anchor patches
    p.add_argument("--zoom-anchor",  type=int, default=17,
                   help="OSM zoom for anchor reference tiles (default: 17)")
    p.add_argument("--zoom-global",  type=int, default=14,
                   help="OSM zoom for global overview tile (default: 14)")
    p.add_argument("--patch-size",   type=int, default=512,
                   help="Anchor patch size in pixels (default: 512)")

    # Orange path
    p.add_argument("--hue-lo1",  type=int, default=5,
                   help="Orange hue lower bound, band 1 (default: 5)")
    p.add_argument("--hue-hi1",  type=int, default=25,
                   help="Orange hue upper bound, band 1 (default: 25)")
    p.add_argument("--hue-lo2",  type=int, default=160,
                   help="Red-orange wrap lower bound (default: 160)")
    p.add_argument("--hue-hi2",  type=int, default=180,
                   help="Red-orange wrap upper bound (default: 180)")
    p.add_argument("--sat-lo",   type=int, default=100,
                   help="Min saturation for orange detection (default: 100)")
    p.add_argument("--val-lo",   type=int, default=80,
                   help="Min brightness for orange detection (default: 80)")
    p.add_argument("--close-px", type=int, default=15,
                   help="Closing radius (px) to bridge dashes (default: 15)")
    p.add_argument("--subsample", type=int, default=500,
                   help="Target skeleton waypoint count (default: 500)")

    # OSRM
    p.add_argument("--osrm-url", default="http://router.project-osrm.org",
                   help="OSRM server URL (default: http://router.project-osrm.org)")
    p.add_argument("--profile",  default="foot",
                   choices=["foot", "bike", "car"],
                   help="OSRM routing profile (default: foot)")
    p.add_argument("--radius",   type=int, default=50,
                   help="OSRM snapping radius in metres (default: 50)")

    # Stage skipping
    p.add_argument("--skip-loftr", action="store_true",
                   help="Skip stage 6 (LoFTR matching); TPS uses geocoded CPs only")
    p.add_argument("--skip-osrm",  action="store_true",
                   help="Skip OSRM map matching; write raw TPS GPX")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}")
        return 1

    py = sys.executable
    stem = image_path.stem

    # Derived output directories
    root       = Path(args.output_dir)
    ocr_dir    = root / "ocr"
    geo_dir    = root / "geocoding"
    osm_dir    = root / "osm"
    anchor_dir = root / "anchors"
    loftr_dir  = root / "loftr"
    tps_dir    = root / "tps"
    path_dir   = root / "path_extract"
    gpx_dir    = root / "gpx"

    # Derived file paths between stages
    ocr_json    = ocr_dir  / f"{stem}_ocr_detections.json"
    geo_json    = geo_dir  / f"{stem}_geocode_candidates.json"
    anchor_json = anchor_dir / f"{stem}_anchors.json"
    loftr_json  = loftr_dir  / f"{stem}_loftr_matches.json"
    tps_json    = tps_dir    / f"{stem}_tps_model.json"
    skel_json   = path_dir   / f"{stem}_skeleton_path.json"

    print(f"[PIPELINE] image: {image_path}")
    print(f"[PIPELINE] output root: {root.resolve()}")

    # ------------------------------------------------------------------
    # Stage 1: OCR
    # ------------------------------------------------------------------
    _run([py, _script("01_ocr.py"),
          "--image",          str(image_path),
          "--output-dir",     str(ocr_dir),
          "--lang",           args.lang,
          "--min-confidence", str(args.min_confidence),
         ], "Stage 1 — OCR")

    if not ocr_json.exists():
        print(f"[ERROR] OCR output not found: {ocr_json}")
        return 1

    # ------------------------------------------------------------------
    # Stage 2: Geocoding
    # ------------------------------------------------------------------
    _run([py, _script("02_geo_coding.py"),
          "--ocr-json",   str(ocr_json),
          "--output-dir", str(geo_dir),
          "--top-k",      str(args.geocode_top_k),
         ], "Stage 2 — Geocoding")

    if not geo_json.exists():
        print(f"[ERROR] geocoding output not found: {geo_json}")
        return 1

    # Verify we have at least one geocoded candidate before proceeding
    geo_data  = json.loads(geo_json.read_text(encoding="utf-8"))
    geocoded  = [c for c in geo_data.get("geocode_candidates", [])
                 if c.get("geocoded")]
    if not geocoded:
        print(
            f"[ERROR] no geocoded candidates — cannot build anchor patches or TPS.\n"
            f"  Possible fixes:\n"
            f"    • Lower --min-confidence (currently {args.min_confidence})\n"
            f"    • Increase --geocode-top-k (currently {args.geocode_top_k})\n"
            f"    • Check that the OCR whitelist contains relevant Korean place names"
        )
        return 1
    print(f"[PIPELINE] {len(geocoded)} geocoded candidate(s) will become anchor patches.")

    # ------------------------------------------------------------------
    # Stage 3: Global OSM tile (for visual reference, not for matching)
    # ------------------------------------------------------------------
    _run([py, _script("03_osm.py"),
          "--geocode-json",    str(geo_json),
          "--zoom",            str(args.zoom_global),
          "--output-dir",      str(osm_dir),
          "--primary-keyword", args.primary_keyword,
         ], "Stage 3 — Global OSM tile")

    # ------------------------------------------------------------------
    # Stage 5: Semantic anchor patches
    # ------------------------------------------------------------------
    _run([py, _script("05_anchor_patches.py"),
          "--image",        str(image_path),
          "--ocr-json",     str(ocr_json),
          "--geocode-json", str(geo_json),
          "--zoom",         str(args.zoom_anchor),
          "--patch-size",   str(args.patch_size),
          "--output-dir",   str(anchor_dir),
         ], "Stage 5 — Anchor patches")

    if not anchor_json.exists():
        print(f"[ERROR] anchor JSON not found: {anchor_json}")
        return 1

    # ------------------------------------------------------------------
    # Stage 6: LoFTR + MAGSAC++ (optional)
    # ------------------------------------------------------------------
    if not args.skip_loftr:
        try:
            _run([py, _script("06_loftr_match.py"),
                  "--anchors-json", str(anchor_json),
                  "--output-dir",   str(loftr_dir),
                 ], "Stage 6 — LoFTR + MAGSAC++")
        except RuntimeError as exc:
            print(f"[WARN] Stage 6 failed (non-fatal): {exc}")
            print("       Continuing with geocoded control points only.")
            loftr_json = None  # type: ignore[assignment]
    else:
        print("[PIPELINE] Stage 6 (LoFTR) skipped via --skip-loftr.")
        loftr_json = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Stage 7: TPS / affine model
    # ------------------------------------------------------------------
    tps_cmd = [py, _script("07_tps_align.py"),
               "--anchors-json", str(anchor_json),
               "--output-dir",   str(tps_dir),
               ]
    if loftr_json is not None and loftr_json.exists():
        tps_cmd += ["--loftr-json", str(loftr_json)]

    _run(tps_cmd, "Stage 7 — TPS alignment model")

    if not tps_json.exists():
        print(f"[ERROR] TPS model not found: {tps_json}")
        return 1

    # ------------------------------------------------------------------
    # Stage 8: Orange path extraction
    # ------------------------------------------------------------------
    _run([py, _script("08_path_extract.py"),
          "--image",      str(image_path),
          "--hue-lo1",    str(args.hue_lo1),
          "--hue-hi1",    str(args.hue_hi1),
          "--hue-lo2",    str(args.hue_lo2),
          "--hue-hi2",    str(args.hue_hi2),
          "--sat-lo",     str(args.sat_lo),
          "--val-lo",     str(args.val_lo),
          "--close-px",   str(args.close_px),
          "--subsample",  str(args.subsample),
          "--output-dir", str(path_dir),
         ], "Stage 8 — Orange path extraction")

    if not skel_json.exists():
        print(f"[ERROR] skeleton path JSON not found: {skel_json}")
        return 1

    # ------------------------------------------------------------------
    # Stage 9: OSRM + GPX
    # ------------------------------------------------------------------
    gpx_cmd = [py, _script("09_osrm_gpx.py"),
               "--skeleton-json", str(skel_json),
               "--tps-json",      str(tps_json),
               "--osrm-url",      args.osrm_url,
               "--profile",       args.profile,
               "--radius",        str(args.radius),
               "--output-dir",    str(gpx_dir),
               ]
    if args.skip_osrm:
        gpx_cmd.append("--skip-osrm")

    _run(gpx_cmd, "Stage 9 — OSRM map matching + GPX")

    gpx_files = list(gpx_dir.glob(f"{stem}*.gpx"))
    if gpx_files:
        print(f"\n{'='*60}")
        print(f"[PIPELINE COMPLETE]")
        print(f"  GPX file: {gpx_files[0]}")
        print(f"{'='*60}")
    else:
        print("[WARN] GPX file not found after stage 9 — check logs above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

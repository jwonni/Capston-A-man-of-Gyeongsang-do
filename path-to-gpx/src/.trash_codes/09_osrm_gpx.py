"""09_osrm_gpx.py — OSRM Map Matching + GPX Generation

Transforms ordered skeleton waypoints from Image A pixel space to real-world
(lat, lon) using the TPS / affine model stored by 07_tps_align.py, then
snaps the GPS trace to the actual road network via the OSRM Match service,
and writes a standard GPX 1.1 track file.

Pipeline
--------
1. Reconstruct the spatial model (TPS or affine) from control points.
2. Project each skeleton waypoint (px, py) → (lat, lon).
3. Validate: discard out-of-range lat/lon values from TPS extrapolation.
4. [Optional] Send waypoints to the OSRM /match/v1 endpoint in chunks.
   - Uses the ``foot`` profile by default so one-way restrictions do not
     block the runner's actual path.
   - Falls back to the raw TPS projection for any failed chunk.
5. Write GPX 1.1 file.

Inputs
------
--skeleton-json   Output of 08_path_extract.py
--tps-json        Output of 07_tps_align.py
--osrm-url        OSRM base URL (default: http://router.project-osrm.org)
--profile         OSRM profile: foot | bike | car  (default: foot)
--radius          Road-snapping search radius in metres (default: 50)
--chunk-size      Waypoints per OSRM HTTP request    (default: 100)
--skip-osrm       Write raw TPS projection as GPX (skip OSRM snapping)
--output-dir      Output directory (default: ./path-to-gpx/output/gpx/)
--output-name     GPX file name stem (default: derived from skeleton JSON)

Outputs
-------
<output-dir>/<name>.gpx
<output-dir>/<name>_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


_OSRM_UA = "MarathonGPXPipeline/1.0"


# ---------------------------------------------------------------------------
# Spatial model reconstruction
# ---------------------------------------------------------------------------

def _load_tps(cps: list[dict], smoothing: float):
    """Reconstruct RBFInterpolator (scipy ≥ 1.7) from stored control points."""
    from scipy.interpolate import RBFInterpolator
    src = np.array([[c["img_x"], c["img_y"]] for c in cps], dtype=np.float64)
    dst = np.array([[c["lat"],   c["lon"]]   for c in cps], dtype=np.float64)
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)


def _load_affine(cps: list[dict]):
    """Reconstruct affine model via least-squares for < 4 control points."""
    src = np.array([[c["img_x"], c["img_y"]] for c in cps], dtype=np.float64)
    dst = np.array([[c["lat"],   c["lon"]]   for c in cps], dtype=np.float64)
    A_src = np.column_stack([src, np.ones(len(src))])
    A, _, _, _ = np.linalg.lstsq(A_src, dst, rcond=None)

    def _predict(query: np.ndarray) -> np.ndarray:
        q_aug = np.column_stack([query, np.ones(len(query))])
        return q_aug @ A

    return _predict


def load_model(tps_json: dict):
    """Return a callable f(query: np.ndarray (N,2)) → np.ndarray (N,2)."""
    cps        = tps_json["control_points"]
    model_type = tps_json.get("model_type", "tps")
    smoothing  = tps_json.get("smoothing", 1e-4)

    if model_type == "tps" and len(cps) >= 4:
        try:
            return _load_tps(cps, smoothing)
        except Exception as exc:
            print(f"  [WARN] TPS reconstruction failed ({exc}), falling back to affine.")

    return _load_affine(cps)


def project_waypoints(
    model,
    waypoints: list[dict],
) -> list[tuple[float, float]]:
    """Apply the spatial model to all waypoints → list of (lat, lon)."""
    pts   = np.array([[w["px"], w["py"]] for w in waypoints], dtype=np.float64)
    latlon = model(pts)  # (N, 2)
    return [(float(row[0]), float(row[1])) for row in latlon]


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def filter_invalid(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop coordinates outside valid lat/lon ranges (TPS extrapolation artefacts)."""
    return [(lat, lon) for lat, lon in coords if _valid(lat, lon)]


# ---------------------------------------------------------------------------
# OSRM Map Matching
# ---------------------------------------------------------------------------

def _osrm_url(
    base: str,
    profile: str,
    coords: list[tuple[float, float]],
    radius: int,
) -> str:
    """Build OSRM /match/v1 request URL for a coordinate chunk."""
    coord_str  = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
    radii_str  = ";".join(str(radius) for _ in coords)
    qs = urllib.parse.urlencode({
        "overview":   "full",
        "geometries": "geojson",
        "radiuses":   radii_str,
        "gaps":       "split",
        "tidy":       "true",
    })
    return f"{base.rstrip('/')}/match/v1/{profile}/{coord_str}?{qs}"


def _match_chunk(
    base: str,
    profile: str,
    coords: list[tuple[float, float]],
    radius: int,
) -> list[tuple[float, float]] | None:
    """Send one OSRM match request; return snapped (lat, lon) list or None."""
    url = _osrm_url(base, profile, coords, radius)
    req = urllib.request.Request(url, headers={"User-Agent": _OSRM_UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"      [WARN] OSRM HTTP error: {exc}")
        return None

    if data.get("code") != "Ok":
        print(f"      [WARN] OSRM code={data.get('code')}: {data.get('message', '')}")
        return None

    snapped: list[tuple[float, float]] = []
    for matching in data.get("matchings", []):
        geom = matching.get("geometry", {})
        for lon_lat in geom.get("coordinates", []):
            snapped.append((lon_lat[1], lon_lat[0]))  # GPX wants lat first

    return snapped or None


def osrm_match_all(
    base: str,
    profile: str,
    coords: list[tuple[float, float]],
    radius: int,
    chunk_size: int,
) -> tuple[list[tuple[float, float]], bool]:
    """
    Match all coordinates, chunked to stay within OSRM's per-request limit.
    Returns (result_coords, all_chunks_ok).
    """
    out: list[tuple[float, float]] = []
    n_chunks = math.ceil(len(coords) / chunk_size)
    all_ok   = True

    for i in range(n_chunks):
        chunk  = coords[i * chunk_size:(i + 1) * chunk_size]
        suffix = f"chunk {i+1}/{n_chunks} ({len(chunk)} pts)"
        print(f"    OSRM {suffix} …", end=" ", flush=True)

        snapped = _match_chunk(base, profile, chunk, radius)
        if snapped:
            print(f"ok → {len(snapped)} pts snapped")
            out.extend(snapped)
        else:
            print("failed → raw projection used")
            out.extend(chunk)
            all_ok = False

    return out, all_ok


# ---------------------------------------------------------------------------
# GPX serialisation
# ---------------------------------------------------------------------------

def write_gpx(
    coords: list[tuple[float, float]],
    out_path: Path,
    name: str = "Marathon Course",
) -> None:
    """Write a minimal but valid GPX 1.1 track file."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="MarathonGPXPipeline"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1'
        ' http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <trk>",
        f"    <name>{name}</name>",
        "    <trkseg>",
    ]
    for lat, lon in coords:
        lines.append(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}"><ele>0</ele></trkpt>')
    lines += ["    </trkseg>", "  </trk>", "</gpx>"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Project skeleton waypoints via TPS → lat/lon, snap to roads via OSRM, write GPX."
    )
    p.add_argument("--skeleton-json", required=True,
                   help="Skeleton path JSON from 08_path_extract.py")
    p.add_argument("--tps-json", required=True,
                   help="TPS / affine model JSON from 07_tps_align.py")
    p.add_argument("--osrm-url", default="http://router.project-osrm.org",
                   help="OSRM base URL (default: http://router.project-osrm.org)")
    p.add_argument("--profile", default="foot", choices=["foot", "bike", "car"],
                   help="OSRM routing profile (default: foot)")
    p.add_argument("--radius", type=int, default=50,
                   help="Road-snapping search radius in metres (default: 50)")
    p.add_argument("--chunk-size", type=int, default=100,
                   help="Waypoints per OSRM request (default: 100)")
    p.add_argument("--skip-osrm", action="store_true",
                   help="Skip OSRM snapping; output raw TPS projection as GPX")
    p.add_argument("--output-dir", default="./path-to-gpx/output/gpx/",
                   help="Output directory (default: ./path-to-gpx/output/gpx/)")
    p.add_argument("--output-name", default=None,
                   help="GPX file name stem (default: derived from skeleton JSON)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    skel_path = Path(args.skeleton_json)
    tps_path  = Path(args.tps_json)

    for path, name in [(skel_path, "--skeleton-json"), (tps_path, "--tps-json")]:
        if not path.exists():
            print(f"[ERROR] {name} file not found: {path}")
            return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skel_data = json.loads(skel_path.read_text(encoding="utf-8"))
    tps_data  = json.loads(tps_path.read_text(encoding="utf-8"))

    stem = args.output_name or Path(skel_data["image"]).stem
    waypoints = skel_data["waypoints"]
    n_cps     = tps_data.get("n_total_cps", 0)

    print(f"[INFO] skeleton waypoints : {len(waypoints)}")
    print(f"[INFO] TPS control points : {n_cps}  ({tps_data.get('model_type','?')})")
    print(f"[INFO] coverage quality   : {tps_data.get('coverage_quality', {})}")

    if n_cps < 2:
        print("[ERROR] TPS model has < 2 control points — cannot project coordinates.")
        return 1

    # ---- TPS projection ------------------------------------------------------
    print("[INFO] projecting waypoints via spatial model …")
    model     = load_model(tps_data)
    raw_ll    = project_waypoints(model, waypoints)
    valid_ll  = filter_invalid(raw_ll)
    dropped   = len(raw_ll) - len(valid_ll)
    print(f"  projected={len(raw_ll)}  valid={len(valid_ll)}  dropped={dropped} (out-of-range)")

    if len(valid_ll) < 10:
        print(
            "[ERROR] fewer than 10 valid lat/lon values after projection.\n"
            "  The TPS model may not cover the course area — check control-point coverage.\n"
            "  Self-residual: "
            f"{tps_data.get('tps_self_residual_mean_deg', 'n/a')}°"
        )
        return 1

    # ---- OSRM map matching ---------------------------------------------------
    osrm_used = False
    if args.skip_osrm:
        print("[INFO] --skip-osrm set; skipping road snapping.")
        final_coords = valid_ll
    else:
        print(f"[INFO] OSRM map matching  profile={args.profile}  "
              f"radius={args.radius}m  url={args.osrm_url}")
        final_coords, all_ok = osrm_match_all(
            args.osrm_url, args.profile, valid_ll,
            args.radius, args.chunk_size,
        )
        osrm_used = True
        if not all_ok:
            print("  [WARN] some OSRM chunks failed; those segments use raw projection.")
        print(f"  final track points after snapping: {len(final_coords)}")

    # ---- GPX output ----------------------------------------------------------
    gpx_path = output_dir / f"{stem}.gpx"
    write_gpx(final_coords, gpx_path, name=stem.replace("_", " "))
    print(f"\n[DONE] GPX written: {gpx_path}  ({len(final_coords)} trackpoints)")

    # ---- Report --------------------------------------------------------------
    report = {
        "image":                skel_data["image"],
        "skeleton_waypoints":   len(waypoints),
        "tps_model_type":       tps_data.get("model_type"),
        "tps_control_points":   n_cps,
        "tps_self_residual_deg": tps_data.get("tps_self_residual_mean_deg"),
        "coverage_quality":     tps_data.get("coverage_quality"),
        "projected_raw":        len(raw_ll),
        "projected_valid":      len(valid_ll),
        "osrm_used":            osrm_used,
        "osrm_url":             args.osrm_url if osrm_used else None,
        "osrm_profile":         args.profile,
        "final_trackpoints":    len(final_coords),
        "gpx_path":             str(gpx_path),
    }
    report_path = output_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
픽셀 ↔ 지리좌표 변환 호모그래피 및 마커 오프셋 캘리브레이션.

앵커 튜플 형식: (pixel_x, pixel_y, lat, lng, text, place)
"""
from __future__ import annotations

import cv2
import numpy as np


class HomographyTransform:
    """정규화 기반 픽셀↔지리좌표 변환기."""

    def __init__(self, anchors: list[tuple]):
        pixel_pts = np.array([[a[0], a[1]] for a in anchors], dtype=np.float64)
        geo_pts   = np.array([[a[3], a[2]] for a in anchors], dtype=np.float64)  # (lng, lat)

        self.px_mean  = pixel_pts.mean(axis=0)
        self.px_std   = pixel_pts.std(axis=0)  + 1e-9
        self.geo_mean = geo_pts.mean(axis=0)
        self.geo_std  = geo_pts.std(axis=0)    + 1e-9

        pn = (pixel_pts - self.px_mean) / self.px_std
        gn = (geo_pts   - self.geo_mean) / self.geo_std

        self.H,     _ = cv2.findHomography(pn.astype(np.float32), gn.astype(np.float32), method=0)
        self.H_inv, _ = cv2.findHomography(gn.astype(np.float32), pn.astype(np.float32), method=0)

    def pixel_to_geo(self, px: float, py: float) -> tuple[float, float]:
        """픽셀 좌표 → (lat, lng)"""
        pn  = np.array([[[(px - self.px_mean[0]) / self.px_std[0],
                          (py - self.px_mean[1]) / self.px_std[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pn, self.H)
        lng = float(res[0][0][0]) * self.geo_std[0] + self.geo_mean[0]
        lat = float(res[0][0][1]) * self.geo_std[1] + self.geo_mean[1]
        return lat, lng

    def geo_to_pixel(self, lat: float, lng: float) -> tuple[float, float]:
        """(lat, lng) → 픽셀 좌표"""
        gn  = np.array([[[(lng - self.geo_mean[0]) / self.geo_std[0],
                          (lat - self.geo_mean[1]) / self.geo_std[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(gn, self.H_inv)
        px  = float(res[0][0][0]) * self.px_std[0] + self.px_mean[0]
        py  = float(res[0][0][1]) * self.px_std[1] + self.px_mean[1]
        return px, py

    def reprojection_error(self, a: tuple) -> float:
        ex, ey = self.geo_to_pixel(a[2], a[3])
        return float(np.sqrt((a[0] - ex) ** 2 + (a[1] - ey) ** 2))

    def reprojection_errors(self, anchors: list[tuple]) -> list[float]:
        return [self.reprojection_error(a) for a in anchors]

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리로 변환."""
        return {
            "px_mean":  self.px_mean.tolist(),
            "px_std":   self.px_std.tolist(),
            "geo_mean": self.geo_mean.tolist(),
            "geo_std":  self.geo_std.tolist(),
            "H":        self.H.tolist(),
            "H_inv":    self.H_inv.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HomographyTransform":
        """딕셔너리에서 복원."""
        obj = cls.__new__(cls)
        obj.px_mean  = np.array(d["px_mean"])
        obj.px_std   = np.array(d["px_std"])
        obj.geo_mean = np.array(d["geo_mean"])
        obj.geo_std  = np.array(d["geo_std"])
        obj.H        = np.array(d["H"])
        obj.H_inv    = np.array(d["H_inv"])
        return obj


def mad_outlier_removal(anchors: list[tuple], thresh: float = 3.0) -> list[tuple]:
    """MAD 기반 위도/경도 분포 이상치 제거."""
    if len(anchors) < 4:
        return anchors
    lats = np.array([a[2] for a in anchors])
    lngs = np.array([a[3] for a in anchors])
    lat_med, lng_med = np.median(lats), np.median(lngs)
    lat_mad = np.median(np.abs(lats - lat_med)) + 1e-9
    lng_mad = np.median(np.abs(lngs - lng_med)) + 1e-9
    return [
        a for a in anchors
        if abs(a[2] - lat_med) / lat_mad <= thresh
        and abs(a[3] - lng_med) / lng_mad <= thresh
    ]


def iterative_outlier_removal(
    anchors: list[tuple],
    max_error_px: float,
    min_anchors: int,
) -> list[tuple]:
    """재투영 오차 기반 반복적 이상치 제거."""
    current = list(anchors)
    while len(current) > min_anchors:
        tf = HomographyTransform(current)
        errors = tf.reprojection_errors(current)
        if max(errors) <= max_error_px:
            break
        current.pop(errors.index(max(errors)))
    return current


# ── 독립 함수 버전 호모그래피 유틸 ───────────────────────────────────────────

def _recompute_homography(anchors: list[tuple]) -> tuple:
    """앵커 목록에서 정규화 호모그래피 행렬과 파라미터를 반환한다."""
    pixel_pts = np.array([[a[0], a[1]] for a in anchors], dtype=np.float64)
    geo_pts   = np.column_stack([[a[3] for a in anchors], [a[2] for a in anchors]])

    px_mean,  px_std  = pixel_pts.mean(axis=0), pixel_pts.std(axis=0) + 1e-9
    geo_mean, geo_std = geo_pts.mean(axis=0),   geo_pts.std(axis=0)   + 1e-9

    pn = (pixel_pts - px_mean) / px_std
    gn = (geo_pts   - geo_mean) / geo_std

    H,     _ = cv2.findHomography(pn.astype(np.float32), gn.astype(np.float32), method=0)
    H_inv, _ = cv2.findHomography(gn.astype(np.float32), pn.astype(np.float32), method=0)

    return H, H_inv, px_mean, px_std, geo_mean, geo_std


def compute_reprojection_errors(anchors: list[tuple]) -> list[float]:
    """각 앵커의 재투영 오차(픽셀) 목록 반환 (독립 함수 버전)."""
    H, H_inv, px_mean, px_std, geo_mean, geo_std = _recompute_homography(anchors)
    errors: list[float] = []
    for a in anchors:
        gn  = np.array([[[(a[3] - geo_mean[0]) / geo_std[0],
                          (a[2] - geo_mean[1]) / geo_std[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(gn, H_inv)
        ex  = float(res[0][0][0]) * px_std[0] + px_mean[0]
        ey  = float(res[0][0][1]) * px_std[1] + px_mean[1]
        errors.append(float(np.sqrt((a[0] - ex) ** 2 + (a[1] - ey) ** 2)))
    return errors


# ── 마커 검출 및 오프셋 캘리브레이션 ─────────────────────────────────────────

def detect_marker_center(
    image:      np.ndarray,
    text_cx:    int,
    text_cy:    int,
    roi_size:   int = 40,
    min_radius: int = 6,
    max_radius: int = 14,
    param1:     int = 50,
    param2:     int = 22,
) -> tuple[int, int] | None:
    """텍스트 좌표 주변 ROI에서 Hough Circle로 마커 중심을 검출한다.

    Args:
        image:    BGR numpy 배열
        text_cx/cy: 텍스트 중심 좌표 (픽셀)
        roi_size: 검색 ROI 반경 (픽셀)

    Returns:
        마커 중심 (x, y) 또는 None
    """
    h, w = image.shape[:2]
    x1 = max(0, text_cx - roi_size)
    y1 = max(0, text_cy - roi_size)
    x2 = min(w, text_cx + roi_size)
    y2 = min(h, text_cy + roi_size)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=30,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    text_local = np.array([text_cx - x1, text_cy - y1])
    best, best_d = None, float("inf")
    for cx, cy, r in circles:
        d = float(np.hypot(cx - text_local[0], cy - text_local[1]))
        if d < best_d and d < roi_size:
            best_d, best = d, (cx, cy)

    if best is None:
        return None
    return (best[0] + x1, best[1] + y1)


def _visualize_calibration(image: np.ndarray, collected: list) -> None:
    try:
        import matplotlib.pyplot as plt
        vis = image.copy()
        for dx, dy, lbl, tpt, mpt in collected:
            cv2.circle(vis, tpt, 4, (0, 0, 255), -1)
            cv2.circle(vis, mpt, 8, (0, 255, 0), 2)
            cv2.arrowedLine(vis, tpt, mpt, (255, 255, 0), 1)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Calibration: 텍스트(빨강) → 마커(초록)")
        plt.axis("off")
        plt.show()
    except Exception:
        pass


def progressive_offset_calibration(
    anchors:         list[tuple],
    image:           np.ndarray,
    consistency_px:  float = 3.0,
    min_samples:     int   = 2,
    visualize:       bool  = False,
) -> tuple[tuple[float, float], list]:
    """앵커 텍스트 좌표 → 마커 중심 오프셋을 점진적으로 추정한다.

    1. 앵커를 순회하며 Hough Circle로 마커 검출
    2. min_samples개 이상 모이면 일관성 검사 (consistency_px 이내면 확정)
    3. 끝까지 일관성 미확보 시 MAD 정리 후 중앙값 반환

    Args:
        anchors:        (pixel_x, pixel_y, lat, lng, text, place) 목록
        image:          BGR numpy 배열
        consistency_px: 오프셋 일관성 임계값 (픽셀)
        min_samples:    일관성 검사 시작 최소 샘플 수
        visualize:      True면 matplotlib 시각화 (서버 환경에서는 False)

    Returns:
        (offset_dx, offset_dy), collected 목록
    """
    collected: list[tuple] = []

    for a in anchors:
        tx, ty = int(a[0]), int(a[1])
        marker = detect_marker_center(image, tx, ty)
        if marker is None:
            continue
        dx, dy = marker[0] - tx, marker[1] - ty
        collected.append((dx, dy, a[4], (tx, ty), marker))

        if len(collected) >= min_samples:
            dxs    = np.array([c[0] for c in collected])
            dys    = np.array([c[1] for c in collected])
            dx_med = float(np.median(dxs))
            dy_med = float(np.median(dys))
            dists  = np.hypot(dxs - dx_med, dys - dy_med)
            if float(np.max(dists)) <= consistency_px:
                final = (dx_med, dy_med)
                if visualize:
                    _visualize_calibration(image, collected)
                return final, collected

    # 모든 앵커 소진 후에도 일관성 미확보 → MAD 정리
    if len(collected) >= 2:
        dxs    = np.array([c[0] for c in collected])
        dys    = np.array([c[1] for c in collected])
        dx_med = float(np.median(dxs))
        dy_med = float(np.median(dys))
        dx_mad = np.median(np.abs(dxs - dx_med)) + 1e-9
        dy_mad = np.median(np.abs(dys - dy_med)) + 1e-9
        mask   = (
            (np.abs(dxs - dx_med) / dx_mad < 3.0) &
            (np.abs(dys - dy_med) / dy_mad < 3.0)
        )
        if int(mask.sum()) >= 2:
            final = (float(np.median(dxs[mask])), float(np.median(dys[mask])))
            kept  = [c for c, m in zip(collected, mask) if m]
            if visualize:
                _visualize_calibration(image, kept)
            return final, kept

    return (0.0, 0.0), collected

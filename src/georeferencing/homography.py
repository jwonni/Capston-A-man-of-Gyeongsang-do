"""
픽셀 ↔ 지리좌표 변환 호모그래피.

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

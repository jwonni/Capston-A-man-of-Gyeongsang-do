"""
경로 스켈레톤화 + 잔가지(spur) 제거.

skimage.morphology.skeletonize 로 1픽셀 너비 중심선을 만든 뒤,
branch point 기반의 graph 분석으로 짧은 잔가지를 반복 제거한다.

알고리즘 (잔가지 탐지):
  1. Branch point(8-연결 이웃 수 ≥ 3) 탐색
  2. Branch point를 일시 제거 → 나머지를 CC로 분리 → 각 segment 획득
  3. Segment가 branch point에 정확히 1개 인접 + 길이 < spur_length → spur
  4. Spur 제거 → branch point 재계산 → 반복

Public API:
    skeletonize_mask(mask, spur_length) → (H, W) uint8
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize as _skimage_skeletonize


def skeletonize_mask(
    mask: np.ndarray,
    spur_length: int = 20,
    morph_close_size: int = 0,
) -> np.ndarray:
    """
    Binary mask를 1픽셀 너비 스켈레톤으로 변환하고 잔가지를 제거한다.

    Args:
        mask            : (H, W) uint8, 경로=255 배경=0
        spur_length     : 이 픽셀 수 미만인 가지를 spur로 간주해 제거.
                          0이면 제거하지 않는다.
        morph_close_size: 스켈레톤화 전 morphology closing 커널 크기.
                          경로를 두껍게 만들어 skeleton 연속성이 향상된다.
                          0이면 비활성화.

    Returns:
        skeleton (H, W) uint8, 스켈레톤=255 배경=0
    """
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skel = _skimage_skeletonize(mask > 127)

    if spur_length > 0:
        skel = _prune_spurs(skel, spur_length)

    return skel.astype(np.uint8) * 255


# ── Internal ──────────────────────────────────────────────────────────────────

def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """각 픽셀의 8-연결 이웃 수를 반환한다."""
    u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(u8, ddepth=-1, kernel=kernel)


def _prune_spurs(skel: np.ndarray, spur_length: int) -> np.ndarray:
    """
    Branch point를 기준으로 skeleton을 segment로 분리한 뒤,
    branch point에 정확히 1개 인접하고 길이 < spur_length 인
    segment(잔가지)를 반복 제거한다.

    n_bp == 1 조건:
      - 0이면 독립 경로(가지 없음) → 절대 제거하지 않는다
      - 1이면 한쪽 끝이 endpoint = 잔가지 → 제거 대상
      - 2이면 두 branch point를 잇는 교량 → 제거하지 않는다
    """
    pruned = skel.copy()
    dilate_k = np.ones((3, 3), dtype=np.uint8)

    for _ in range(500):
        nc = _neighbor_count(pruned)
        branch_pts = pruned & (nc >= 3)
        segments_only = pruned & ~branch_pts

        num_labels, labels = cv2.connectedComponents(
            segments_only.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            break

        removed = False
        for lbl in range(1, num_labels):
            seg = labels == lbl
            if int(seg.sum()) >= spur_length:
                continue

            dilated = cv2.dilate(seg.astype(np.uint8), dilate_k) > 0
            n_bp = int(np.sum(branch_pts & dilated & ~seg))

            if n_bp == 1:
                pruned[seg] = False
                removed = True

        if not removed:
            break

    return pruned

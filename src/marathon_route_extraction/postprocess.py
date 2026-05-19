"""
마라톤 경로 후처리 파이프라인.

이 모듈은 top-level `postprocess/` 폴더에 있던 파이프라인을
프로젝트 내부로 완전히 내장한 버전이다.

Step 1. Connected component 탐색 + 주경로 선택
Step 2. 형태 기반 노이즈 제거
Step 3. 끊어진 경로 조각 연결
Step 4. 잔여 fragment 제거
Step 5. 스켈레톤화 + 잔가지 제거
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize as _skimage_skeletonize, thin as _skimage_thin
from src.config import (
    AREA_THRESH,
    CIRC_THRESH,
    SKEL_THRESH,
    MAX_DISTANCE,
    MIN_FRAGMENT_SIZE,
    LINE_THICKNESS,
    MORPH_CLOSE_SIZE,
    FINAL_SIZE_THRESH,
    SPUR_LENGTH,
    SKEL_MORPH_CLOSE,
)


def load_mask(mask_path: str | Path) -> np.ndarray:
    """PNG 마스크를 bool 배열로 로드한다."""
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return arr > 127


def postprocess_mask(
    mask: np.ndarray,
    area_thresh: int = AREA_THRESH,                # 이 픽셀 수 미만이면 노이즈 후보
    circ_thresh: float = CIRC_THRESH,              # circularity 초과 시 원형으로 간주 (0~1)
    skel_thresh: int = SKEL_THRESH,                # skeleton 길이(px) 미만이면 노이즈 후보
    max_distance: float = MAX_DISTANCE,            # fragment 연결 허용 최대 거리 (px)
    min_fragment_size: int = MIN_FRAGMENT_SIZE,    # 연결 대상 fragment 최소 픽셀 수
    line_thickness: int = LINE_THICKNESS,          # 연결선 두께 (px)
    morph_close_size: Optional[int] = MORPH_CLOSE_SIZE, # morphology closing 커널 크기. 미지정 시 입력 mask 단변의 1%%(최소 5)로 자동 계산
    final_size_thresh: int = FINAL_SIZE_THRESH,    # Step 3 후 남은 fragment 중 이 픽셀 수 미만을 제거. 0=주경로만 보존.
    spur_length: int = SPUR_LENGTH,                # 이 픽셀 수 미만인 가지를 잔가지로 간주해 제거. 0=제거 안 함.
    skel_morph_close: int = SKEL_MORPH_CLOSE,      # 스켈레톤화 전 morphology closing 커널 크기 (0=비활성화, 권장 3~7).
) -> np.ndarray:
    """
    후처리 파이프라인 진입점.

    변경: marathon-path-seg의 `postprocess_mask`와 동일하게
    전체 단계 결과를 생성/반환하도록 동작을 맞춥니다.
    이유: 중간 결과(binary mask, skeleton, contour, endpoint 등)를
    단계별로 추적/검증하기 위해 최종 스켈레톤만 반환하던 이전 동작을 확장합니다.

    Returns (same order as marathon-path-seg pipeline):
        main_mask, noise_mask, filtered_mask, connected_mask,
        final_mask, skeleton_mask, features, noise_labels, connect_log
    """
    # morph_close_size 해석은 pipeline 기준과 동일하게 처리
    resolved_morph_close = _resolve_morph_close_size(mask.shape, morph_close_size)
    return run_pipeline(
        mask,
        area_thresh=area_thresh,
        circ_thresh=circ_thresh,
        skel_thresh=skel_thresh,
        max_distance=max_distance,
        min_fragment_size=min_fragment_size,
        line_thickness=line_thickness,
        morph_close_size=resolved_morph_close,
        final_size_thresh=final_size_thresh,
        spur_length=spur_length,
        skel_morph_close=skel_morph_close,
        verbose=False,
    )


def run_pipeline(
    mask_uint8: np.ndarray,
    area_thresh: int = AREA_THRESH,
    circ_thresh: float = CIRC_THRESH,
    skel_thresh: int = SKEL_THRESH,
    max_distance: float = MAX_DISTANCE,
    min_fragment_size: int = MIN_FRAGMENT_SIZE,
    line_thickness: int = LINE_THICKNESS,
    morph_close_size: Optional[int] = None,
    final_size_thresh: int = FINAL_SIZE_THRESH,
    spur_length: int = SPUR_LENGTH,
    skel_morph_close: int = SKEL_MORPH_CLOSE,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, List, List]:
    """전체 후처리 파이프라인을 실행하고 각 단계 결과를 반환한다."""
    _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    resolved_morph_close = _resolve_morph_close_size(mask_uint8.shape, morph_close_size)

    bool_mask = mask_uint8 > 127
    labeled_mask, components = find_connected_components(bool_mask)
    main_mask, main_comp = select_main_path(labeled_mask, components)

    if verbose:
        total = sum(c["size"] for c in components)
        main_px = main_comp.get("size", 0)
        ratio = main_px / total * 100 if total > 0 else 0
        print(f"  [Step 1] {len(components)}개 연결 요소 → 주경로 {main_px:,} px ({ratio:.1f}%)")

    filtered_mask, noise_mask, features, noise_labels = filter_noise(
        mask_uint8,
        area_thresh=area_thresh,
        circ_thresh=circ_thresh,
        skel_thresh=skel_thresh,
    )

    if verbose:
        kept = len(features) - len(noise_labels)
        print(f"  [Step 2] 노이즈 {len(noise_labels)}개 제거 / {kept}개 보존")
        _print_feature_table(features, noise_labels)

    if verbose:
        print("  [Step 3] Fragment 연결 시작...")

    connected_mask, connect_log = iterative_fragment_connection(
        filtered_mask,
        max_distance=max_distance,
        min_fragment_size=min_fragment_size,
        line_thickness=line_thickness,
        morph_close_size=resolved_morph_close,
        verbose=verbose,
    )

    if verbose:
        print("  [Step 4] 잔여 fragment 제거 시작...")

    final_mask = remove_small_fragments(connected_mask, min_size=final_size_thresh)

    if verbose:
        num_conn, _, _, _ = find_components(connected_mask)
        num_clean, _, _, _ = find_components(final_mask)
        removed = (num_conn - 1) - (num_clean - 1)
        print(f"  [Step 4] 잔여 fragment {removed}개 제거 → {num_clean - 1}개")

    if verbose:
        label = f"spur < {spur_length} px" if spur_length > 0 else "spur 제거 비활성화"
        close_label = f"morph_close={skel_morph_close}px" if skel_morph_close > 0 else "morph_close 비활성화"
        print(f"  [Step 5] 스켈레톤화 시작... ({label} / {close_label})")

    skeleton_mask = skeletonize_mask(
        final_mask,
        spur_length=spur_length,
        morph_close_size=skel_morph_close,
    )

    if verbose:
        skel_px = int((skeleton_mask > 0).sum())
        print(f"  [Step 5] 스켈레톤 완료 → {skel_px:,} px")

    return main_mask, noise_mask, filtered_mask, connected_mask, final_mask, skeleton_mask, features, noise_labels, connect_log


def find_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """이진 마스크에서 4-연결성 connected components를 탐색한다."""
    labeled_mask, num_labels = _label_components(binary_mask)
    counts = np.bincount(labeled_mask.ravel())
    components = [
        {"label": lbl, "size": int(counts[lbl])}
        for lbl in range(1, num_labels + 1)
    ]
    components.sort(key=lambda x: x["size"], reverse=True)
    for rank, comp in enumerate(components, start=1):
        comp["rank"] = rank
    return labeled_mask, components


def select_main_path(labeled_mask: np.ndarray, components: List[dict]) -> Tuple[np.ndarray, dict]:
    """가장 큰 component를 주경로로 선택한다."""
    if not components:
        return np.zeros(labeled_mask.shape, dtype=bool), {}
    main_comp = components[0]
    return labeled_mask == main_comp["label"], main_comp


def filter_noise(
    mask: np.ndarray,
    area_thresh: int,
    circ_thresh: float,
    skel_thresh: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, dict], List[int]]:
    """형태 기반 노이즈 제거를 수행한다."""
    num, labels, stats, _ = _cv2_components(mask)
    if num <= 1:
        return mask.copy(), np.zeros_like(mask), {}, []

    main_label = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])

    features: Dict[int, dict] = {}
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        width = int(stats[lbl, cv2.CC_STAT_WIDTH])
        height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        bbox_area = max(width * height, 1)
        extent = float(area / bbox_area)
        short_side = max(min(width, height), 1)
        bbox_aspect = float(max(width, height) / short_side)
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        skel_len = int(_skimage_skeletonize(comp_mask > 0).sum())
        features[lbl] = {
            "label": lbl,
            "area": area,
            "circularity": round(_circularity(area, comp_mask), 4),
            "skeleton_length": skel_len,
            "bbox_width": width,
            "bbox_height": height,
            "bbox_area": bbox_area,
            "extent": round(extent, 4),
            "bbox_aspect_ratio": round(bbox_aspect, 4),
            "noise_score": 0,
            "noise_reasons": [],
            "is_main": lbl == main_label,
        }

    extent_thresh = 0.45
    path_like_aspect_thresh = 2.5
    noise_labels: List[int] = []
    for lbl, feat in features.items():
        if feat["is_main"]:
            continue
        path_like = feat["bbox_aspect_ratio"] >= path_like_aspect_thresh
        score, reasons = 0, []
        if feat["area"] < area_thresh:
            score += 2
            reasons.append("area")
        if feat["circularity"] > circ_thresh:
            score += 1
            reasons.append("circularity")
        if feat["extent"] > extent_thresh and not path_like:
            score += 1
            reasons.append("extent")
        if feat["skeleton_length"] < skel_thresh and feat["extent"] > extent_thresh and not path_like:
            score += 1
            reasons.append("skeleton")
        feat["noise_score"] = score
        feat["noise_reasons"] = reasons
        if score >= 2:
            noise_labels.append(lbl)

    filtered_mask = mask.copy()
    noise_mask = np.zeros_like(mask)
    for lbl in noise_labels:
        region = labels == lbl
        noise_mask[region] = 255
        filtered_mask[region] = 0

    return filtered_mask, noise_mask, features, noise_labels


def find_components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """8-연결성 기준 connected components 계산."""
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def iterative_fragment_connection(
    initial_mask: np.ndarray,
    max_distance: float = 50.0,
    min_fragment_size: int = 10,
    line_thickness: int = 2,
    morph_close_size: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[dict]]:
    """Main path와 fragment를 반복적으로 연결한다."""
    mask = initial_mask.copy()

    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if verbose:
            print(f"  [Morphology Close] kernel={morph_close_size}px")

    log: List[dict] = []

    for iteration in range(1, 10_000):
        num_labels, labels, stats, _ = find_components(mask)
        main_label, frag_labels = _split_main_fragments(num_labels, stats)

        if main_label is None:
            break

        frag_labels = [
            lbl for lbl in frag_labels
            if stats[lbl, cv2.CC_STAT_AREA] >= min_fragment_size
        ]
        if not frag_labels:
            if verbose:
                area = int(stats[main_label, cv2.CC_STAT_AREA])
                print(f"  [Iter {iteration}] 연결 가능한 fragment 없음. Main: {area:,} px. 종료.")
            break

        main_mask = (labels == main_label).astype(np.uint8) * 255
        main_dt, main_dt_labels, main_dt_coords = _build_main_distance_transform(main_mask)
        if main_dt is None or len(main_dt_coords) == 0:
            break

        best: dict = {"dist": float("inf")}
        for frag_lbl in frag_labels:
            frag_mask = (labels == frag_lbl).astype(np.uint8) * 255
            frag_pt, main_pt, dist = _find_distance_transform_connection(
                main_dt,
                main_dt_labels,
                main_dt_coords,
                frag_mask,
            )
            if main_pt is not None and dist < best["dist"]:
                best = {
                    "dist": dist,
                    "label": frag_lbl,
                    "area": int(stats[frag_lbl, cv2.CC_STAT_AREA]),
                    "main_pt": main_pt,
                    "frag_pt": frag_pt,
                    "frag_mask": frag_mask,
                }

        if best["dist"] > max_distance:
            if verbose:
                print(f"  [Iter {iteration}] 최근접 fragment 거리 {best['dist']:.1f} px > {max_distance} px. 종료.")
            break

        mask[best["frag_mask"] > 0] = 255
        _draw_polyline_bridge(mask, best["main_pt"], best["frag_pt"], line_thickness)
        log.append({
            "iteration": iteration,
            "fragment_label": int(best["label"]),
            "fragment_area_px": best["area"],
            "distance_px": round(best["dist"], 2),
            "main_point": best["main_pt"].tolist(),
            "frag_point": best["frag_pt"].tolist(),
        })
        if verbose:
            print(f"  [Iter {iteration:>3}] Fragment #{best['label']:>4} 연결 — 크기: {best['area']:>6,} px, 거리: {best['dist']:>7.1f} px")

    return mask, log


def remove_small_fragments(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """주경로를 제외한 fragment 중 min_size 미만인 것을 제거한다."""
    num_labels, labels, stats, _ = find_components(mask)
    if num_labels <= 1:
        return mask.copy()

    main_label = max(range(1, num_labels), key=lambda lbl: int(stats[lbl, cv2.CC_STAT_AREA]))
    cleaned = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        if lbl == main_label:
            cleaned[labels == lbl] = 255
        elif min_size > 0 and int(stats[lbl, cv2.CC_STAT_AREA]) >= min_size:
            cleaned[labels == lbl] = 255
    return cleaned


def skeletonize_mask(
    mask: np.ndarray,
    spur_length: int = 20,
    morph_close_size: int = 0,
) -> np.ndarray:
    """Binary mask를 1픽셀 너비 스켈레톤으로 변환하고 잔가지를 제거한다."""
    if morph_close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skel = _skimage_skeletonize(mask > 127)
    skel = _skimage_thin(skel, max_num_iter=1)

    if spur_length > 0:
        skel = _prune_spurs(skel, spur_length)

    return skel.astype(np.uint8) * 255


def _resolve_morph_close_size(mask_shape: Tuple[int, int], morph_close_size: Optional[int]) -> int:
    if morph_close_size is not None:
        return morph_close_size
    short_side = min(mask_shape[:2])
    return max(int(round(short_side * 0.01)), 5)


def _print_feature_table(features: Dict, noise_labels: List) -> None:
    noise_set = set(noise_labels)
    print(
        f"  {'레이블':>6}  {'크기(px)':>9}  {'원형도':>8}  {'스켈레톤':>8}  {'extent':>7}  {'aspect':>7}  {'점수':>4}  {'결과':>6}"
    )
    for feat in sorted(features.values(), key=lambda x: x["area"], reverse=True):
        decision = "noise" if feat["label"] in noise_set else "keep "
        marker = " ← 제거" if feat["label"] in noise_set else ""
        role = "MAIN" if feat["is_main"] else "    "
        print(
            f"  {feat['label']:>6}  {feat['area']:>9,}  {feat['circularity']:>8.4f}  {feat['skeleton_length']:>8}  "
            f"{feat.get('extent', 0.0):>7.2f}  {feat.get('bbox_aspect_ratio', 0.0):>7.2f}  {feat.get('noise_score', 0):>4}  "
            f"{role} {decision}{marker}"
        )


def _label_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label as scipy_label

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        labeled, n = scipy_label(binary_mask, structure=structure)
        return labeled.astype(np.int32), int(n)
    except ImportError:
        return _label_bfs(binary_mask)


def _label_bfs(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = binary_mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    path_ys, path_xs = np.where(binary_mask)

    for i in range(len(path_ys)):
        y, x = int(path_ys[i]), int(path_xs[i])
        if labeled[y, x] != 0:
            continue
        current_label += 1
        queue = deque([(y, x)])
        labeled[y, x] = current_label
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and binary_mask[ny, nx] and labeled[ny, nx] == 0:
                    labeled[ny, nx] = current_label
                    queue.append((ny, nx))

    return labeled, current_label


def _cv2_components(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _circularity(area: int, comp_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return float(4.0 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0


def _split_main_fragments(num_labels: int, stats: np.ndarray) -> Tuple[Optional[int], List[int]]:
    if num_labels <= 1:
        return None, []
    areas = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[0][0], [lbl for lbl, _ in areas[1:]]


def _extract_boundary_points(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.empty((0, 2), dtype=np.int64)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = binary & (cv2.erode(binary, kernel, iterations=1) == 0)
    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([rows, cols])


def _build_main_distance_transform(
    main_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    binary = (main_mask > 127).astype(np.uint8)
    if int(binary.sum()) == 0:
        return None, None, np.empty((0, 2), dtype=np.int64)

    src = (binary == 0).astype(np.uint8)
    main_dt, main_dt_labels = cv2.distanceTransformWithLabels(
        src,
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    main_dt_coords = np.column_stack(np.where(binary > 0))
    return main_dt, main_dt_labels, main_dt_coords


def _find_distance_transform_connection(
    main_dt: np.ndarray,
    main_dt_labels: np.ndarray,
    main_dt_coords: np.ndarray,
    frag_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    frag_pts = _extract_boundary_points(frag_mask)
    if len(frag_pts) == 0:
        rows, cols = np.where(frag_mask > 0)
        if len(rows) == 0:
            return None, None, float("inf")
        frag_pts = np.column_stack([rows, cols])

    dists = main_dt[frag_pts[:, 0], frag_pts[:, 1]]
    best_idx = int(np.argmin(dists))
    frag_pt = frag_pts[best_idx]

    label = int(main_dt_labels[frag_pt[0], frag_pt[1]])
    main_idx = label - 1
    if main_idx < 0 or main_idx >= len(main_dt_coords):
        return None, None, float("inf")

    main_pt = main_dt_coords[main_idx]
    return frag_pt, main_pt, float(dists[best_idx])


def _draw_polyline_bridge(
    mask: np.ndarray,
    main_pt: np.ndarray,
    frag_pt: np.ndarray,
    line_thickness: int,
) -> None:
    mid_pt = np.rint((main_pt.astype(np.float32) + frag_pt.astype(np.float32)) / 2.0).astype(np.int32)
    pts = np.array(
        [
            [int(main_pt[1]), int(main_pt[0])],
            [int(mid_pt[1]), int(mid_pt[0])],
            [int(frag_pt[1]), int(frag_pt[0])],
        ],
        dtype=np.int32,
    )
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(u8, ddepth=-1, kernel=kernel)


def _prune_spurs(skel: np.ndarray, spur_length: int) -> np.ndarray:
    pruned = skel.copy()
    dilate_k = np.ones((3, 3), dtype=np.uint8)

    for _ in range(500):
        nc = _neighbor_count(pruned)
        branch_pts = pruned & (nc >= 3)
        segments_only = pruned & ~branch_pts
        num_labels, labels = cv2.connectedComponents(segments_only.astype(np.uint8), connectivity=8)
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
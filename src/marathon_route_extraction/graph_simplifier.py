"""
Graph simplification pipeline for skeleton-based route extraction.

5단계 그래프 구조 단순화

    1. 1차 선형 압축  degree-2 중간 노드 제거
    2. 노드 병합      – 유클리드 거리 ≤ τ 인 노드를 centroid로 통합
    3. 2차 선형 압축  – 병합 후 새로 생긴 degree-2 노드 재압축
    4. 컴포넌트 연결  – 분리 성분을 가장 가까운 노드쌍으로 반복 연결
    5. 핵심 노드 유지 – leaf(degree 1) + junction(degree ≥ 3) 만 남은 G'

Public API
----------
simplify_graph(skeleton, tau)                            → Graph
find_nearest_node(graph, cy, cx)                         → (row, col) | None
restore_detailed_path(graph, key_nodes)                  → [(row, col), ...]
extract_keypoints(pixels, graph, angle_thresh, min_dist) → [(row, col), ...]

타입 정의
---------
Graph = dict[ (row,col) , dict[ (row,col) , {"pixels": [(row,col), ...]} ] ]
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional

import numpy as np

# ── 타입 별칭 ─────────────────────────────────────────────────────────────────
Graph = dict[tuple[int, int], dict[tuple[int, int], dict]]

_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1),
              (0,  -1),           (0,  1),
              (1,  -1), (1,  0), (1,  1)]


# ══════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _skeleton_to_graph(skeleton: np.ndarray) -> Graph:
    """
    스켈레톤 픽셀 하나 = 노드 하나.
    8-connected 인접 픽셀끼리 엣지 연결.
    엣지 속성 'pixels' 에 [시작, 끝] 초기값을 저장한다.
    """
    points = [tuple(int(v) for v in p) for p in np.argwhere(skeleton)]
    point_set = set(points)
    graph: Graph = {p: {} for p in points}
    for y, x in points:
        for dy, dx in _OFFSETS_8:
            nb = (y + dy, x + dx)
            if nb in point_set:
                graph[(y, x)][nb] = {"pixels": [(y, x), nb]}
    return graph


def _degree(graph: Graph, node: tuple) -> int:
    return len(graph.get(node, {}))


# ══════════════════════════════════════════════════════════════════════════════
# 1 & 3단계 — 선형 경로 압축
# ══════════════════════════════════════════════════════════════════════════════

def _compress_linear_chains(graph: Graph) -> Graph:
    """
    degree-2 노드가 연속된 선형 구간 (v1, ..., vk) 을 찾아
    v1 ↔ vk 단일 엣지로 대체한다.
    중간 노드들의 픽셀 좌표는 엣지 속성 'pixels' 로 보존된다.
    """
    visited_edges: set[frozenset] = set()
    new_graph: Graph = {n: {} for n in graph}
    original_degree2 = {n for n in graph if _degree(graph, n) == 2}

    for start in list(graph.keys()):
        if _degree(graph, start) == 2:
            continue  # 중간 노드에서는 체인을 시작하지 않는다

        for first_nb in list(graph[start].keys()):
            key = frozenset([start, first_nb])
            if key in visited_edges:
                continue
            visited_edges.add(key)

            # 픽셀 시퀀스 수집
            chain_pixels: list[tuple[int, int]] = list(
                graph[start][first_nb]["pixels"]
            )
            prev, cur = start, first_nb

            while _degree(graph, cur) == 2:
                neighbors = [n for n in graph[cur] if n != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                edge_key = frozenset([cur, nxt])
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)

                seg = graph[cur][nxt]["pixels"]
                if seg[0] == chain_pixels[-1]:
                    chain_pixels.extend(seg[1:])
                else:
                    chain_pixels.extend(reversed(seg[:-1]))

                prev, cur = cur, nxt

            end = cur
            if end == start:
                # 완전한 루프 — 셀프 루프로 보존
                new_graph[start][start] = {"pixels": chain_pixels}
                continue

            new_graph[start][end] = {"pixels": chain_pixels}
            new_graph[end][start] = {"pixels": list(reversed(chain_pixels))}

    # degree-2 였던 노드 중 이웃이 없어진 것은 제거
    cleaned: Graph = {}
    for n, neighbors in new_graph.items():
        if n in original_degree2 and len(neighbors) == 0:
            continue
        cleaned[n] = neighbors

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# 2단계 — 노드 병합
# ══════════════════════════════════════════════════════════════════════════════

def _merge_nearby_nodes(graph: Graph, tau: float) -> Graph:
    """
    유클리드 거리 ≤ τ 인 노드들을 클러스터링하여
    centroid에 가장 가까운 실제 노드를 대표 노드로 통합한다.
    분기점 주변의 픽셀 단위 중복 노드를 제거하는 것이 목적이다.
    """
    nodes = list(graph.keys())
    if not nodes:
        return graph

    node_arr = np.array(nodes, dtype=np.float64)  # (N, 2)

    # Union-Find
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    tau2 = tau * tau
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dy = node_arr[i, 0] - node_arr[j, 0]
            dx = node_arr[i, 1] - node_arr[j, 1]
            if dy * dy + dx * dx <= tau2:
                union(i, j)

    # 클러스터별 대표 노드 결정 (centroid 에 가장 가까운 실제 노드)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(nodes)):
        clusters[find(i)].append(i)

    rep: dict[tuple, tuple] = {}
    for root, members in clusters.items():
        pts = node_arr[members]
        centroid = pts.mean(axis=0)
        dists = ((pts - centroid) ** 2).sum(axis=1)
        best = members[int(np.argmin(dists))]
        rep_node = nodes[best]
        for m in members:
            rep[nodes[m]] = rep_node

    # 새 그래프 구성
    new_graph: Graph = {}
    for node, neighbors in graph.items():
        rn = rep[node]
        if rn not in new_graph:
            new_graph[rn] = {}
        for nb, attr in neighbors.items():
            rnb = rep[nb]
            if rnb == rn:
                continue  # 셀프 루프 제거
            # 더 짧은 엣지 유지
            if rnb not in new_graph[rn]:
                new_graph[rn][rnb] = attr
            else:
                if len(attr["pixels"]) < len(new_graph[rn][rnb]["pixels"]):
                    new_graph[rn][rnb] = attr

    # 역방향 엣지 동기화
    for n in list(new_graph.keys()):
        for nb, attr in list(new_graph[n].items()):
            if nb not in new_graph:
                new_graph[nb] = {}
            if n not in new_graph[nb]:
                new_graph[nb][n] = {"pixels": list(reversed(attr["pixels"]))}

    return new_graph


# ══════════════════════════════════════════════════════════════════════════════
# 4단계 — 컴포넌트 연결
# ══════════════════════════════════════════════════════════════════════════════

def _find_components(graph: Graph) -> list[set[tuple]]:
    """BFS 로 연결 성분 목록 반환."""
    visited: set[tuple] = set()
    components: list[set[tuple]] = []
    for start in graph:
        if start in visited:
            continue
        comp: set[tuple] = set()
        q: deque[tuple] = deque([start])
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in graph[n]:
                if nb not in visited:
                    q.append(nb)
        components.append(comp)
    return components


def _connect_components(graph: Graph) -> Graph:
    """
    분리된 연결 성분을 가장 가까운 노드쌍으로 반복 연결.
    단일 연결 그래프가 될 때까지 반복한다.
    브리지 엣지의 픽셀 시퀀스는 두 노드 사이의 직선 보간으로 채운다.
    """
    while True:
        comps = _find_components(graph)
        if len(comps) <= 1:
            break

        comps.sort(key=len, reverse=True)  # 큰 성분 우선

        best_dist = float("inf")
        best_pair: Optional[tuple[tuple, tuple]] = None

        main_arr = np.array(list(comps[0]), dtype=np.float64)

        for comp in comps[1:]:
            other_arr = np.array(list(comp), dtype=np.float64)
            diff = main_arr[:, None, :] - other_arr[None, :, :]  # (M, N, 2)
            dists = (diff ** 2).sum(axis=2)                       # (M, N)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            d = float(dists[idx])
            if d < best_dist:
                best_dist = d
                n1 = tuple(int(v) for v in main_arr[idx[0]])
                n2 = tuple(int(v) for v in other_arr[idx[1]])
                best_pair = (n1, n2)

        if best_pair is None:
            break

        n1, n2 = best_pair
        r1, c1 = n1
        r2, c2 = n2
        steps = max(abs(r2 - r1), abs(c2 - c1), 1)
        bridge = [
            (int(round(r1 + (r2 - r1) * t / steps)),
             int(round(c1 + (c2 - c1) * t / steps)))
            for t in range(steps + 1)
        ]
        graph[n1][n2] = {"pixels": bridge}
        graph[n2][n1] = {"pixels": list(reversed(bridge))}

    return graph


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def simplify_graph(skeleton: np.ndarray, tau: float = 5.0) -> Graph:
    """
    스켈레톤 이미지를 받아 5단계 단순화를 거친 의미론적 그래프 G' 를 반환한다.

    Args:
        skeleton : (H, W) bool 또는 uint8 배열 — 스켈레톤 픽셀이 True/255
        tau      : 노드 병합 거리 임계값 (픽셀 단위)

    Returns:
        단순화된 그래프 G' (leaf + junction 노드만 포함)
    """
    # 0. 초기 밀집 그래프 구성
    graph = _skeleton_to_graph(skeleton)
    if not graph:
        return graph

    # 1단계: 1차 선형 압축
    graph = _compress_linear_chains(graph)

    # 2단계: 노드 병합
    graph = _merge_nearby_nodes(graph, tau)

    # 3단계: 2차 선형 압축
    graph = _compress_linear_chains(graph)

    # 4단계: 컴포넌트 연결
    graph = _connect_components(graph)

    # 5단계: leaf + junction 만 남은 상태 (위 과정의 자연스러운 결과)
    return graph


def find_nearest_node(
    graph: Graph,
    cy: int,
    cx: int,
) -> Optional[tuple[int, int]]:
    """
    단순화된 그래프 G' 의 노드 중 (cy=row, cx=col) 에 가장 가까운 노드를 반환.

    Args:
        graph : simplify_graph() 의 반환값
        cy    : 기준점의 row (y)
        cx    : 기준점의 col (x)
    """
    nodes = list(graph.keys())
    if not nodes:
        return None
    arr = np.array(nodes, dtype=np.float64)
    dists = (arr[:, 0] - cy) ** 2 + (arr[:, 1] - cx) ** 2
    idx = int(np.argmin(dists))
    return nodes[idx]


def restore_detailed_path(
    graph: Graph,
    key_nodes: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    G' 위의 핵심 노드 시퀀스를 받아 각 엣지의 pixels 속성을 이어붙여
    원본 스켈레톤 픽셀 연속 경로를 복원한다.

    Args:
        graph     : simplify_graph() 의 반환값
        key_nodes : BFS 등으로 구한 핵심 노드 순서 리스트

    Returns:
        (row, col) 튜플의 연속 픽셀 경로
    """
    detailed: list[tuple[int, int]] = []
    for i in range(len(key_nodes) - 1):
        n1, n2 = key_nodes[i], key_nodes[i + 1]
        seg = graph[n1].get(n2, {}).get("pixels", [n1, n2])
        if detailed and seg and seg[0] == detailed[-1]:
            detailed.extend(seg[1:])
        else:
            detailed.extend(seg)
    return detailed


def extract_keypoints(
    pixels: list[tuple[int, int]],
    graph: Graph,
    angle_thresh: float = 20.0,
    min_dist: float = 8.0,
) -> list[tuple[int, int]]:
    """
    픽셀 시퀀스에서 의미 있는 좌표만 추출한다.

    두 가지 기준으로 추출:
        1. leaf(degree=1) / junction(degree>=3) 노드 → 무조건 포함
        2. 연속된 세 픽셀의 방향 변화(각도)가 angle_thresh 이상 → 꺾임점으로 추출

    Args:
        pixels       : restore_detailed_path() 의 반환값 [(row, col), ...]
        graph        : simplify_graph() 의 반환값 (leaf/junction 판별용)
        angle_thresh : 꺾임으로 판단할 최소 각도 (도 단위, 기본 20.0)
        min_dist     : 연속 키포인트 간 최소 거리 (픽셀 단위, 기본 8.0)

    Returns:
        (row, col) 튜플의 키포인트 리스트
    """
    if len(pixels) < 3:
        return list(pixels)

    # leaf / junction 노드 좌표 수집 — 무조건 포함 대상
    must_include: set[tuple[int, int]] = set()
    for n, neighbors in graph.items():
        d = len(neighbors)
        if d == 1 or d >= 3:
            must_include.add(n)

    keypoints: list[tuple[int, int]] = [pixels[0]]

    for i in range(1, len(pixels) - 1):
        p_prev = pixels[i - 1]
        p_cur  = pixels[i]
        p_next = pixels[i + 1]

        # leaf / junction 이면 각도 무관하게 무조건 추가
        if p_cur in must_include:
            # 바로 직전에 추가된 점과 너무 가까우면 교체 (중복 방지)
            if keypoints[-1] != p_cur:
                keypoints.append(p_cur)
            continue

        # 방향 벡터 계산
        v1 = (p_cur[0] - p_prev[0], p_cur[1] - p_prev[1])
        v2 = (p_next[0] - p_cur[0],  p_next[1] - p_cur[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])

        if len1 < 1e-8 or len2 < 1e-8:
            continue

        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))

        if angle > angle_thresh:
            last = keypoints[-1]
            dist = math.hypot(p_cur[0] - last[0], p_cur[1] - last[1])
            if dist >= min_dist:
                keypoints.append(p_cur)

    # 끝점 무조건 포함
    if keypoints[-1] != pixels[-1]:
        keypoints.append(pixels[-1])

    return keypoints

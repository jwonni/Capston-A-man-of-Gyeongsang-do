"""
Zhang-Suen thinning 과정 시각화 스크립트.

목표:
- 마스크가 skeleton으로 얇아지는 과정을 단계별로 눈으로 확인한다.
- README에 올리기 좋은 정적 요약 이미지와 GIF를 함께 만든다.

출력물:
1) 단계별 프레임 PNG들
2) 프레임 묶음 요약 이미지(contact sheet)
3) 애니메이션 GIF
4) 단계별 foreground 픽셀 수 JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("src", None)


def build_arg_parser() -> argparse.ArgumentParser:
    """커맨드라인 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="Visualize Zhang-Suen thinning process for a binary mask."
    )
    parser.add_argument(
        "--input-mask",
        type=str,
        required=True,
        help="Path to input mask image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/thinning_visualization",
        help="Directory to save frames, GIF, and summary image.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Foreground threshold for grayscale mask binarization.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20,
        help="Remove connected components smaller than this area.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Maximum number of frames included in summary image and GIF.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=240,
        help="Duration per frame in output GIF (milliseconds).",
    )
    return parser


def load_binary_mask(mask_path: Path, threshold: int) -> np.ndarray:
    """마스크 이미지를 읽고 이진 배열(bool)로 변환한다."""
    mask = Image.open(mask_path).convert("L")
    mask_arr = np.asarray(mask, dtype=np.uint8)
    return mask_arr > threshold


def connected_components(binary: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-연결 성분을 추출한다."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component: List[Tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

        components.append(component)

    return components


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """작은 노이즈 성분을 제거한다."""
    if min_area <= 0:
        return binary.copy()

    cleaned = np.zeros_like(binary, dtype=bool)
    for component in connected_components(binary):
        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[np.array(ys), np.array(xs)] = True
    return cleaned


def _zhang_suen_step(img: np.ndarray, step: int) -> List[Tuple[int, int]]:
    """Zhang-Suen의 한 sub-step에서 제거할 픽셀 목록을 찾는다."""
    padded = np.pad(img, 1, mode="constant")
    to_remove: List[Tuple[int, int]] = []

    # foreground 픽셀만 순회하여 제거 조건을 검사한다.
    foreground_points = np.argwhere(img > 0)
    for y, x in foreground_points:
        py, px = y + 1, x + 1

        p2 = int(padded[py - 1, px])
        p3 = int(padded[py - 1, px + 1])
        p4 = int(padded[py, px + 1])
        p5 = int(padded[py + 1, px + 1])
        p6 = int(padded[py + 1, px])
        p7 = int(padded[py + 1, px - 1])
        p8 = int(padded[py, px - 1])
        p9 = int(padded[py - 1, px - 1])

        # B(p): 이웃 foreground 수
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        if b < 2 or b > 6:
            continue

        # A(p): 0->1 전이 횟수
        neighbors = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
        a = sum((neighbors[i] == 0 and neighbors[i + 1] == 1) for i in range(8))
        if a != 1:
            continue

        if step == 0:
            if p2 * p4 * p6 != 0:
                continue
            if p4 * p6 * p8 != 0:
                continue
        else:
            if p2 * p4 * p8 != 0:
                continue
            if p2 * p6 * p8 != 0:
                continue

        to_remove.append((y, x))

    return to_remove


def zhang_suen_with_history(binary: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """Thinning 결과와 단계별 상태(history)를 반환한다.

    history는 각 반복/sub-step 이후의 이미지를 누적한다.
    pixel_counts는 history와 같은 길이로 foreground 픽셀 수를 담는다.
    """
    img = binary.astype(np.uint8).copy()
    history: List[np.ndarray] = [img.astype(bool).copy()]
    pixel_counts: List[int] = [int(np.sum(img))]

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = _zhang_suen_step(img, step=step)
            if not to_remove:
                continue

            changed = True
            for y, x in to_remove:
                img[y, x] = 0

            # sub-step이 적용될 때마다 상태를 기록한다.
            history.append(img.astype(bool).copy())
            pixel_counts.append(int(np.sum(img)))

    return img.astype(bool), history, pixel_counts


def bool_to_rgb_image(binary: np.ndarray) -> Image.Image:
    """bool 마스크를 RGB 이미지로 변환한다."""
    gray = binary.astype(np.uint8) * 255
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def annotate_frame(image: Image.Image, title: str) -> Image.Image:
    """프레임 상단에 단계 정보 텍스트를 넣는다."""
    top_margin = 26
    canvas = Image.new("RGB", (image.width, image.height + top_margin), (255, 255, 255))
    canvas.paste(image, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def sample_history_indices(total_steps: int, max_frames: int) -> List[int]:
    """히스토리에서 균등 샘플링할 index 목록을 만든다."""
    if total_steps <= max_frames:
        return list(range(total_steps))

    # 0과 마지막 step은 반드시 포함하고, 그 사이를 균등 간격으로 선택한다.
    sampled = np.linspace(0, total_steps - 1, num=max_frames)
    indices = sorted(set(int(round(value)) for value in sampled.tolist()))
    if indices[0] != 0:
        indices[0] = 0
    if indices[-1] != total_steps - 1:
        indices[-1] = total_steps - 1
    return indices


def save_frames(
    history: Sequence[np.ndarray],
    pixel_counts: Sequence[int],
    output_dir: Path,
    stem: str,
) -> List[Path]:
    """모든 history 프레임을 PNG로 저장한다."""
    frames_dir = output_dir / f"{stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []

    for step_index, frame in enumerate(history):
        frame_image = bool_to_rgb_image(frame)
        title = f"step={step_index:03d} | foreground_pixels={pixel_counts[step_index]}"
        annotated = annotate_frame(frame_image, title)
        frame_path = frames_dir / f"frame_{step_index:03d}.png"
        annotated.save(frame_path)
        frame_paths.append(frame_path)

    return frame_paths


def make_contact_sheet(images: Sequence[Image.Image], cols: int = 4) -> Image.Image:
    """이미지 여러 장을 grid로 묶는다."""
    if not images:
        raise ValueError("No images provided for contact sheet")

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * max_width, rows * max_height), (245, 245, 245))

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        sheet.paste(image, (col * max_width, row * max_height))

    return sheet


def create_outputs(
    input_mask_path: Path,
    output_dir: Path,
    cleaned_mask: np.ndarray,
    history: Sequence[np.ndarray],
    pixel_counts: Sequence[int],
    max_frames: int,
    gif_duration_ms: int,
) -> None:
    """README용 시각화 결과를 생성하고 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_mask_path.stem

    # 1) 단계별 프레임 전체 저장
    frame_paths = save_frames(history, pixel_counts, output_dir, stem)

    # 2) README용 요약 프레임 선택
    sampled_indices = sample_history_indices(total_steps=len(history), max_frames=max_frames)
    sampled_images: List[Image.Image] = []
    for index in sampled_indices:
        frame = bool_to_rgb_image(history[index])
        title = f"step={index:03d} | pixels={pixel_counts[index]}"
        sampled_images.append(annotate_frame(frame, title))

    # 첫 패널에 cleaned input을 명시적으로 추가해 시작 상태를 더 분명히 보여준다.
    cleaned_panel = annotate_frame(
        bool_to_rgb_image(cleaned_mask),
        f"cleaned input | pixels={int(np.sum(cleaned_mask))}",
    )
    if sampled_images and sampled_indices[0] == 0:
        sampled_images[0] = cleaned_panel
    else:
        sampled_images.insert(0, cleaned_panel)

    contact_sheet = make_contact_sheet(sampled_images, cols=4)
    contact_sheet_path = output_dir / f"{stem}_thinning_process_sheet.png"
    contact_sheet.save(contact_sheet_path)

    # 3) GIF 저장
    gif_path = output_dir / f"{stem}_thinning_process.gif"
    sampled_images[0].save(
        gif_path,
        save_all=True,
        append_images=sampled_images[1:],
        duration=max(40, gif_duration_ms),
        loop=0,
        optimize=False,
    )

    # 4) 단계별 픽셀 수 기록(JSON)
    stats_path = output_dir / f"{stem}_thinning_stats.json"
    payload = {
        "input_mask": str(input_mask_path),
        "total_steps": len(history) - 1,
        "initial_foreground_pixels": int(pixel_counts[0]),
        "final_foreground_pixels": int(pixel_counts[-1]),
        "sampled_indices": sampled_indices,
        "pixel_counts": [int(value) for value in pixel_counts],
        "all_frame_paths": [str(path) for path in frame_paths],
        "sheet_path": str(contact_sheet_path),
        "gif_path": str(gif_path),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"All frames saved to: {output_dir / f'{stem}_frames'}")
    print(f"Summary sheet saved to: {contact_sheet_path}")
    print(f"GIF saved to: {gif_path}")
    print(f"Stats JSON saved to: {stats_path}")


def main() -> None:
    """CLI 진입점."""
    args = build_arg_parser().parse_args()

    input_mask_path = Path(args.input_mask)
    if not input_mask_path.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask_path}")

    output_dir = Path(args.output_dir)

    # 원본 마스크를 이진화하고 작은 노이즈를 먼저 제거한다.
    binary_mask = load_binary_mask(input_mask_path, threshold=args.threshold)
    cleaned_mask = remove_small_components(binary_mask, min_area=args.min_component_area)

    # thinning 전체 과정을 기록하여 단계별 변화를 시각화할 수 있게 한다.
    _, history, pixel_counts = zhang_suen_with_history(cleaned_mask)

    create_outputs(
        input_mask_path=input_mask_path,
        output_dir=output_dir,
        cleaned_mask=cleaned_mask,
        history=history,
        pixel_counts=pixel_counts,
        max_frames=max(4, args.max_frames),
        gif_duration_ms=args.gif_duration_ms,
    )


if __name__ == "__main__":
    main()
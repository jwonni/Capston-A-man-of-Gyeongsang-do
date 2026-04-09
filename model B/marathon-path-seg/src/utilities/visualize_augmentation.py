"""
데이터 증강 시각화 스크립트

train 이미지에 대해 augmentation이 어떻게 적용되는지 시각적으로 보여준다.
- 원본 이미지/마스크
- 다양한 augmentation이 적용된 버전 (이미지/마스크 쌍)
을 matplotlib grid로 시각화한다.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

# 프로젝트 root 추가
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.augmentation import apply_basic_augmentation
from src.core.utils import collect_pairs


def visualize_augmentation_samples(
    image_path: Path,
    mask_path: Path,
    num_augmented: int = 6,
    image_size: int = 256,
) -> None:
    """
    한 장의 이미지-마스크 쌍에 대해 여러 augmentation을 적용하고 시각화한다.

    Args:
        image_path: 이미지 파일 경로
        mask_path: 마스크 파일 경로
        num_augmented: 시각화할 augmented 버전 개수
        image_size: 리사이징할 크기
    """
    # 원본 로드
    orig_image = Image.open(image_path).convert("RGB")
    orig_mask = Image.open(mask_path).convert("L")

    # 리사이징
    orig_image_resized = orig_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    orig_mask_resized = orig_mask.resize((image_size, image_size), Image.Resampling.NEAREST)

    # Augmented 버전 생성
    augmented_images = []
    augmented_masks = []
    for _ in range(num_augmented):
        aug_img, aug_mask = apply_basic_augmentation(orig_image.copy(), orig_mask.copy())
        aug_img_resized = aug_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        aug_mask_resized = aug_mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        augmented_images.append(aug_img_resized)
        augmented_masks.append(aug_mask_resized)

    # 시각화: 상단 행 = 원본, 이어서 augmented 이미지들
    # 하단 행 = 각각의 마스크
    fig, axes = plt.subplots(2, num_augmented + 1, figsize=(18, 6))
    fig.suptitle(f"Data Augmentation Visualization\n{image_path.name}", fontsize=14, fontweight="bold")

    # 원본 이미지와 마스크
    axes[0, 0].imshow(orig_image_resized)
    axes[0, 0].set_title("Original Image", fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(orig_mask_resized, cmap="gray")
    axes[1, 0].set_title("Original Mask", fontweight="bold")
    axes[1, 0].axis("off")

    # Augmented 버전들
    for i, (aug_img, aug_mask) in enumerate(zip(augmented_images, augmented_masks)):
        col = i + 1
        axes[0, col].imshow(aug_img)
        axes[0, col].set_title(f"Augmented #{i+1}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(aug_mask, cmap="gray")
        axes[1, col].set_title(f"Mask #{i+1}", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    # plt.savefig("augmentation_visualization.png", dpi=100, bbox_inches="tight")
    # print(f"✓ Visualization saved to augmentation_visualization.png")
    plt.show()


def main() -> None:
    """
    첫 번째 train 이미지를 선택하여 augmentation을 시각화한다.
    """
    print("=" * 80)
    print("Data Augmentation Visualization")
    print("=" * 80)

    data_root = Path("data/train")
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"❌ Data directory not found at {data_root}")
        print(f"   Expected: {images_dir} and {masks_dir}")
        return

    # 데이터셋 쌍 수집
    pairs = collect_pairs(images_dir, masks_dir)
    if not pairs:
        print("❌ No image-mask pairs found!")
        return

    image_path, mask_path = pairs[0]
    print(f"✓ Sample image: {image_path.name}")
    print(f"✓ Sample mask:  {mask_path.name}")
    print(f"\n이미지에 다음 증강이 적용됩니다 (확률 기반):")
    print(f"  - 좌우 뒤집기    (50% 확률)")
    print(f"  - 상하 뒤집기    (20% 확률)")
    print(f"  - 회전 (±10도)  (30% 확률)")
    print(f"\n각 증강은 이미지와 마스크에 동일하게 적용됩니다.")
    print(f"시각화를 생성 중입니다...\n")

    visualize_augmentation_samples(
        image_path,
        mask_path,
        num_augmented=6,
        image_size=256,
    )

    print("\n" + "=" * 80)
    print("✓ 완료! augmentation_visualization.png를 확인하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()

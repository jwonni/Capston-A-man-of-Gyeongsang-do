"""
데이터 증강 모듈

마라톤 경로 분할 학습용으로 가벼운 기하학적 증강을 제공한다.
- 좌우 뒤집기
- 상하 뒤집기
- 작은 각도 회전
"""

import random
from typing import Tuple

from PIL import Image, ImageOps


def apply_basic_augmentation(
    image: Image.Image,
    mask: Image.Image,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.2,
    rotate_prob: float = 0.3,
    max_rotate_deg: float = 10.0,
) -> Tuple[Image.Image, Image.Image]:
    """
    이미지-마스크 쌍에 동일한 랜덤 증강을 적용한다.

    Args:
        image: RGB PIL 이미지
        mask: L(단일 채널) PIL 마스크
        hflip_prob: 좌우 뒤집기 확률
        vflip_prob: 상하 뒤집기 확률
        rotate_prob: 회전 적용 확률
        max_rotate_deg: 회전 최대 각도 ([-max, +max])

    Returns:
        증강된 (image, mask)
    """
    if random.random() < hflip_prob:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    if random.random() < vflip_prob:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    if random.random() < rotate_prob:
        angle = random.uniform(-max_rotate_deg, max_rotate_deg)

        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )
        mask = mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )

    return image, mask

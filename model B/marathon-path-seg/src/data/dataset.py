"""
데이터셋 모듈
마라톤 경로 분할 데이터셋 정의

각 이미지와 마스크 쌍을 불러와서 모델에 필요한 형태로 전처리하는 기능을 제공한다.
- 이미지: RGB로 열고, 지정된 크기로 조절한 후, 모델별로 필요한 전처리를 적용하여 (C, H, W) 텐서로 반환
- 마스크: 흑백으로 열고, 지정된 크기로 조절한 후, 이진화하여 (1, H, W) 텐서로 반환

U-Net과 DeepLabV3 모델 모두에서 사용할 수 있도록 유연한 전처리 기능을 제공한다.
U-Net은 [0, 1] 범위의 스케일링만 적용하고, DeepLabV3는 ImageNet 평균과 표준편차로 정규화한다.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentation import apply_basic_augmentation


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def apply_model_preprocess(image_tensor: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    모델별 입력 전처리를 적용한다.

    Args:
        image_tensor: (C, H, W) 범위 [0, 1] 이미지 텐서
        model_name: 모델 이름

    Returns:
        전처리된 (C, H, W) 텐서
    """
    name = model_name.lower()

    if name == "deeplabv3":
        mean = IMAGENET_MEAN.to(device=image_tensor.device, dtype=image_tensor.dtype)
        std = IMAGENET_STD.to(device=image_tensor.device, dtype=image_tensor.dtype)
        return (image_tensor - mean) / std

    # 기본값: U-Net/기타 모델은 [0, 1] 스케일만 사용
    return image_tensor


class MarathonSegDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        image_size: int,
        model_name: str = "unet",
        use_augmentation: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.model_name = model_name
        self.use_augmentation = use_augmentation

    def __len__(self) -> int:
        return len(self.pairs)

    # 이미지와 마스크를 불러와서 전처리하는 함수
    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        
        # 이미지와 마스크를 열고, 각각 RGB와 L(단일 채널) 모드로 변환한다.
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.use_augmentation:
            image, mask = apply_basic_augmentation(image, mask)

        # 이미지 크기 조절 시 BILINEAR 보간법을 사용하여 픽셀 사이를 부드럽게 채움
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # 마스크 크기 조절 시 NEAREST(최근접 이웃) 보간법을 사용하여 픽셀 값을 그대로 유지
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        # PIL 이미지를 numpy 배열로 변환
        # 이미지 배열은 0 ~ 255 사이의 픽셀 값을 0.0 ~ 1.0 사이로 정규화
        # 마스크 배열은 0 ~ 255 사이의 픽셀 값을 0과 1로 이진화 (127을 기준으로)하여 float32 타입으로 변환
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        # 이미지 배열을 텐서로 변환하고, (H, W, C) -> (C, H, W) 순서로 변경
        # 모델별로 필요한 전처리를 적용한 후, 배치 차원을 추가하여 (1, C, H, W) 형태로 반환
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        image_tensor = apply_model_preprocess(image_tensor, self.model_name)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return image_tensor, mask_tensor

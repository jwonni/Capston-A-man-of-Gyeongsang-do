"""
유틸리티 모듈
공통 함수들과 헬퍼 함수들 정의
"""

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def set_seed(seed: int) -> None:
    """
    난수 생성 시드를 고정하여 실험의 재현성을 보장하는 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 마라톤 이미지와 마스킹 이미지 짝짓기 함수 
def collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """
    이미지와 마스크 파일을 짝짓는 함수
    
    Args:
        images_dir: 이미지 파일이 있는 디렉토리 경로
        masks_dir: 마스크 파일이 있는 디렉토리 경로
    
    Returns:
        (이미지_경로, 마스크_경로) 튜플의 리스트
    """
    # 이미지와 마스크가 짝지어진 리스트를 저장할 빈 리스트 생성
    pairs: List[Tuple[Path, Path]] = [] 
    # 이미지 폴더에서 유효한 확장자를 가진 파일들만 리스트로 만듬
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]

    # 이미지 파일들을 순서대로 처리하면서, 각 이미지에 대응하는 마스크 파일이 있는지 확인
    for image_path in sorted(image_paths):
        stem = image_path.stem
        candidates = [masks_dir / f"{stem}{ext}" for ext in VALID_EXTENSIONS]
        mask_path = next((p for p in candidates if p.exists()), None)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


# Train과 Validation 데이터 분할 함수
def split_pairs(
    pairs: Sequence[Tuple[Path, Path]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    이미지-마스크 쌍들을 training과 validation 데이터로 분할하는 함수
    
    Args:
        pairs: (이미지_경로, 마스크_경로) 튜플의 리스트
        val_ratio: validation 데이터의 비율 (0.0 ~ 1.0)
        seed: 난수 생성 시드
    
    Returns:
        (train_pairs, val_pairs) 튜플
    """
    pairs = list(pairs)
    rng = random.Random(seed) 
    rng.shuffle(pairs) # 랜덤 시드를 사용하여 데이터를 섞음 - 이렇게 하면 매번 같은 순서로 섞이게 되어, 실험의 재현성을 높여줌
    val_count = max(1, int(len(pairs) * val_ratio)) # 일정 비율로 분할, val 데이터는 최소 하나는 있도록 설정
    val_pairs = pairs[:val_count] # 섞인 리스트에서 앞부분을 validation 데이터로, 나머지를 training 데이터로 사용
    train_pairs = pairs[val_count:] 
    return train_pairs, val_pairs



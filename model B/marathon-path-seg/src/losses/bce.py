"""
BCE (Binary Cross Entropy) 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross Entropy with logits 손실 함수
    극도로 불균형한 클래스에 대해 pos_weight를 적용하여 경로 픽셀에 더 큰 가중치를 부여함
    """
    def __init__(self, pos_weight: float = 10.0) -> None:
        """
        Args:
            pos_weight: 양성(경로) 픽셀에 부여할 가중치.
                       foreground가 매우 희소할 때(현재 데이터처럼) 
                       BCE가 배경 위주로 학습되는 것을 완화함
        """
        super().__init__()
        # pos_weight는 양성(경로) 픽셀을 더 크게 벌주기 위한 가중치
        # foreground가 매우 희소할 때(현재 데이터처럼) BCE가 배경 위주로 학습되는 것을 완화함
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델이 출력한 raw logits (batch_size, 1, H, W)
            target: 정답 마스크 (batch_size, 1, H, W)
        
        Returns:
            BCE 손실값
        """
        # logits와 같은 device/dtype으로 맞춰서 CPU/GPU 혼용 에러를 방지
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )

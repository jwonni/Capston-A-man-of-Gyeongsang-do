"""
Dice (F1 Score) 기반 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    학습용 부드러운 Dice 손실 함수
    threshold을 사용하지 않고 확률값을 그대로 사용하여 미분 가능하게 만듦
    
    Dice = (2 * 교집합) / (합집합)
    """
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class SoftDiceLoss(nn.Module):
    """
    부드러운 Dice 손실 함수
    threshold을 사용하지 않고 확률값을 그대로 사용하여 미분 가능하게 만듦
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델이 출력한 raw logits (batch_size, 1, H, W)
            target: 정답 마스크 (batch_size, 1, H, W)
        
        Returns:
            Dice 손실값
        """
        return soft_dice_loss_from_logits(logits, target)


class BCEDiceLoss(nn.Module):
    """
    BCE와 Soft Dice 손실의 조합
    
    BCE와 Dice를 조합하여 사용하면 두 손실의 장점을 모두 활용할 수 있음:
    - BCE: 픽셀별 분류 성능에 집중
    - Dice: 전체적인 영역 겹침 정도(F1 score)에 집중
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 10.0,
    ) -> None:
        """
        Args:
            bce_weight: BCE 손실의 가중치
            dice_weight: Dice 손실의 가중치
            pos_weight: 양성(경로) 픽셀에 부여할 가중치
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델이 출력한 raw logits (batch_size, 1, H, W)
            target: 정답 마스크 (batch_size, 1, H, W)
        
        Returns:
            BCE와 Dice 손실의 가중 합
        """
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        dice_loss = soft_dice_loss_from_logits(logits, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

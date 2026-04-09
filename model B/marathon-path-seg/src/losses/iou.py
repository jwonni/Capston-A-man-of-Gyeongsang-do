"""
IoU (Intersection over Union) 기반 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Soft IoU loss for training (differentiable). Thresholding을 쓰지 않고 확률값 그대로 사용함.
def soft_iou_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    학습용 부드러운 IoU 손실 함수
    threshold을 사용하지 않고 확률값을 그대로 사용하여 미분 가능하게 만듦
    """
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


class SoftIoULoss(nn.Module):
    """
    부드러운 IoU 손실 함수
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
            IoU 손실값
        """
        return soft_iou_loss_from_logits(logits, target)


class BCEIoULoss(nn.Module):
    """
    BCE와 Soft IoU 손실의 조합
    
    BCE와 IoU를 조합하여 사용하면 두 손실의 장점을 모두 활용할 수 있음:
    - BCE: 픽셀별 분류 성능에 집중
    - IoU: 전체적인 영역 겹침 정도에 집중
    """
    def __init__(
        self,
        # Todo: bce랑 iou의 Loss 비율 조정하면서 성능 향상 시도해보기 - 예를 들어, bce_weight=0.5, iou_weight=0.5로 설정하여 두 손실이 동일한 비중으로 모델 학습에 기여하도록 할 수 있음
        bce_weight: float = 0.5, 
        iou_weight: float = 0.5,

        # Todo: pos_weight 조정하면서 성능 향상 시도해보기 - 예를 들어, pos_weight=10.0으로 설정하여 경로 픽셀의 오차가 배경 픽셀보다 10배 더 크게 반영되도록 할 수 있음 - 이렇게 하면 모델이 경로 픽셀을 더 잘 학습하도록 도와줄 수 있음
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        # pos_weight는 양성(경로) 픽셀을 더 크게 벌주기 위한 가중치
        # foreground가 매우 희소할 때(현재 데이터처럼) BCE가 배경 위주로 학습되는 것을 완화함
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델이 출력한 raw logits (batch_size, 1, H, W)
            target: 정답 마스크 (batch_size, 1, H, W)
        
        Returns:
            BCE와 IoU 손실의 가중 합
        """
        # logits와 같은 device/dtype으로 맞춰서 CPU/GPU 혼용 에러를 방지
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        iou_loss = soft_iou_loss_from_logits(logits, target)
        return self.bce_weight * bce_loss + self.iou_weight * iou_loss

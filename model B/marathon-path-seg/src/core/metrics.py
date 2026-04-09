"""
메트릭 모듈
성능 평가 메트릭 정의
"""

import torch


# Dice score
def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 모델이 출력한 raw logits을 sigmoid를 통해 확률로 변환
    probs = torch.sigmoid(logits)
    # 확률이 0.5보다 크면 1(경로), 작으면 0(배경)으로 이진화
    preds = (probs > 0.5).float()
    
    # 배치 내의 모든 픽셀을 하나의 벡터로 평탄화
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    # Dice 계산: (2 * 교집합) / (합집합)
    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


# IoU (Intersection over Union) score - Dice와 다르게 False Positive/Negative를 더 엄격하게 평가함
# IoU = 교집합 / 합집합 = (TP) / (TP + FP + FN) - Dice와 달리 교집합을 두 배하지 않음
def iou_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 모델이 출력한 raw logits을 sigmoid를 통해 확률로 변환
    probs = torch.sigmoid(logits)
    # 확률이 0.5보다 크면 1(경로), 작으면 0(배경)으로 이진화
    preds = (probs > 0.5).float()

    # 배치 내의 모든 픽셀을 하나의 벡터로 평탄화
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    # IoU 계산: 교집합 / (TP + FP + FN)
    intersection = (preds * target).sum(dim=1)
    # Union = 두 집합의 원소 개수 합 - 교집합 (즉,합집합)
    union = preds.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

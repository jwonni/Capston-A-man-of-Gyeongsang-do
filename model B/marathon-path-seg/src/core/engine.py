"""
학습 엔진 모듈
epoch 실행 로직 정의
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class EpochStats:
    loss: float
    dice: float
    iou: float

# 단일 epoch 동안 모델을 학습 또는 평가하는 함수
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            from .metrics import dice_score_from_logits, iou_score_from_logits

            dice = dice_score_from_logits(logits, masks)
            iou = iou_score_from_logits(logits, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice.item() * batch_size
        running_iou += iou.item() * batch_size
        count += batch_size

    return EpochStats(loss=running_loss / count, dice=running_dice / count, iou=running_iou / count)

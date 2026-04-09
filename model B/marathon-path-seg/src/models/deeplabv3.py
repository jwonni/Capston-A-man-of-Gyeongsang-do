"""
DeepLabV3 모델 래퍼
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models import ResNet50_Weights


class DeepLabV3Model(nn.Module):
	def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
		super().__init__()
		if in_channels != 3:
			raise ValueError("DeepLabV3Model currently supports in_channels=3 only.")

		'''
		weights=None avoids any external download and keeps behavior deterministic.
		
		가중치를 불러오고 싶다면 torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)로 변경할 수 있다.
		backbone 가중치도 None으로 설정하여 완전히 랜덤 초기화된 모델을 사용한다. 필요에 따라 pretrained backbone을 사용할 수도 있다.
		# self.model = deeplabv3_resnet50(weights=None, weights_backbone=None)
		'''
		# ResNet50_Weights.IMAGENET1K_V2는 ImageNet 데이터셋으로 사전 학습된 ResNet50 백본을 사용한다. 필요에 따라 다른 가중치를 사용할 수도 있다.
		
		self.model = deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
		self.model.classifier = DeepLabHead(2048, out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.model(x)["out"]

		# Keep a strict output contract: logits shape must match input HxW.
		if out.shape[-2:] != x.shape[-2:]:
			out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

		return out

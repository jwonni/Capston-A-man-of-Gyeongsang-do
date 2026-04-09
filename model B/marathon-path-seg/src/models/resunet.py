"""
ResUNet model
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		if in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels),
			)
		else:
			self.shortcut = nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.shortcut(x)

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = out + residual
		out = self.relu(out)

		return out


class DownBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()

		self.pool = nn.MaxPool2d(kernel_size=2)
		self.block = ResidualBlock(in_channels, out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(x)
		return self.block(x)


class UpBlock(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()

		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.block = ResidualBlock(out_channels + skip_channels, out_channels)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)

		if x.shape[-2:] != skip.shape[-2:]:
			x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

		x = torch.cat([x, skip], dim=1)
		return self.block(x)


class ResUNet(nn.Module):
	def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
		super().__init__()

		self.enc1 = ResidualBlock(in_channels, base_channels)
		self.enc2 = DownBlock(base_channels, base_channels * 2)
		self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
		self.enc4 = DownBlock(base_channels * 4, base_channels * 8)

		self.bridge = DownBlock(base_channels * 8, base_channels * 16)

		self.dec4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
		self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
		self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
		self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels)

		self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		e1 = self.enc1(x)
		e2 = self.enc2(e1)
		e3 = self.enc3(e2)
		e4 = self.enc4(e3)

		b = self.bridge(e4)

		d4 = self.dec4(b, e4)
		d3 = self.dec3(d4, e3)
		d2 = self.dec2(d3, e2)
		d1 = self.dec1(d2, e1)

		out = self.head(d1)

		if out.shape[-2:] != x.shape[-2:]:
			out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

		return out

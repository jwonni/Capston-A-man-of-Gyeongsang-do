"""
U-Net architecture and inference utilities.
Mirrors the training-time architecture exactly.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.marathon_route_extraction.component_filter import remove_small_components


# ── Morphological helpers ─────────────────────────────────────────────────────


def _erode(binary: np.ndarray) -> np.ndarray:
    """3×3 morphological erosion (pure numpy)."""
    h, w = binary.shape
    p = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        p[0:h, 0:w] & p[0:h, 1:w+1] & p[0:h, 2:w+2] &
        p[1:h+1, 0:w] & p[1:h+1, 1:w+1] & p[1:h+1, 2:w+2] &
        p[2:h+2, 0:w] & p[2:h+2, 1:w+1] & p[2:h+2, 2:w+2]
    )


def _dilate(binary: np.ndarray) -> np.ndarray:
    """3×3 morphological dilation (pure numpy)."""
    h, w = binary.shape
    p = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        p[0:h, 0:w] | p[0:h, 1:w+1] | p[0:h, 2:w+2] |
        p[1:h+1, 0:w] | p[1:h+1, 1:w+1] | p[1:h+1, 2:w+2] |
        p[2:h+2, 0:w] | p[2:h+2, 1:w+1] | p[2:h+2, 2:w+2]
    )


def apply_postprocess(
    binary: np.ndarray,
    opening_iterations: int = 0,
    closing_iterations: int = 0,
    min_component_area: int = 0,
) -> np.ndarray:
    for _ in range(opening_iterations):
        binary = _erode(binary)
    for _ in range(opening_iterations):
        binary = _dilate(binary)
    for _ in range(closing_iterations):
        binary = _dilate(binary)
    for _ in range(closing_iterations):
        binary = _erode(binary)
    return remove_small_components(binary, min_area=min_component_area)


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        ch = base_channels
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch,     ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(ch * 8, ch * 16)
        self.up4  = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        self.up3  = nn.ConvTranspose2d(ch * 8,  ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 8,  ch * 4)
        self.up2  = nn.ConvTranspose2d(ch * 4,  ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 4,  ch * 2)
        self.up1  = nn.ConvTranspose2d(ch * 2,  ch,     kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2,  ch)
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bridge(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ── Loading & inference ───────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    base_channels = args.get("base_channels", 32)
    model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def predict_mask(
    model: UNet,
    image_pil: Image.Image,
    device: torch.device,
    image_size: int = 512,
    threshold: float = 0.5,
    min_component_area: int = 20,
    opening_iterations: int = 0,
    closing_iterations: int = 0,
) -> tuple[Image.Image, Image.Image]:
    orig_size = image_pil.size  # (W, H)

    processed = image_pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(processed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary  = probs > threshold
    cleaned = apply_postprocess(binary, opening_iterations, closing_iterations, min_component_area)

    mask_pil = Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L").resize(
        orig_size, Image.Resampling.NEAREST
    )
    return image_pil, mask_pil
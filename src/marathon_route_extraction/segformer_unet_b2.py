"""
SegFormer-UNet v3 (B2) — 아키텍처 및 추론 유틸리티.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ENCODER_CHANNELS = {"b0": [32, 64, 160, 256], "b2": [64, 128, 320, 512], "b4": [64, 128, 320, 512]}
_DECODER_CHANNELS = {"b0": [128, 64, 32],       "b2": [256, 128, 64],      "b4": [256, 128, 64]}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 기본 블록 ──────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 p: int = 1, d: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class RefinementBlock(nn.Module):
    """Inverted Bottleneck (×4 expand) + Depthwise 3×3 + SE + Residual."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch * 4
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.se(self.body(x)) + self.shortcut(x))


# ── CBAM ──────────────────────────────────────────────────────────────────────

class _ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch: int, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = _ChannelAttention(ch)
        self.sa = _SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── PixelShuffle 출력 헤드 ────────────────────────────────────────────────────

class PixelShuffleHead(nn.Module):
    """H/4 → H/2 → H 2단계 PixelShuffle 업샘플링."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2), ConvBNReLU(in_ch, in_ch),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2), ConvBNReLU(in_ch, in_ch),
        )
        self.head = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.up2(self.up1(x)))


# ── ASPP Bridge ────────────────────────────────────────────────────────────────

class ASPPBridge(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: tuple = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([ConvBNReLU(in_ch, out_ch, k=3, p=r, d=r) for r in rates])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.proj = ConvBNReLU(out_ch * (len(rates) + 1), out_ch, k=1, p=0)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.dropout(self.proj(torch.cat(feats, dim=1)))


# ── FusionNeck ─────────────────────────────────────────────────────────────────

class FusionNeck(nn.Module):
    def __init__(self, encoder_channels: list, embed_dim: int = 256) -> None:
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
            for c in encoder_channels
        ])

    def forward(self, features: list) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        out = None
        for proj, feat in zip(self.projs, features):
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feat = proj(feat)
            out = feat if out is None else out + feat
        return out


# ── _UpBlock ──────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 align_channels: int | None = None, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        ac = align_channels if align_channels is not None else out_channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, ac, kernel_size=1, bias=False),
            nn.BatchNorm2d(ac), nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(ac, spatial_kernel_size)
        self.conv = RefinementBlock(in_channels + ac, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.cbam(self.skip_proj(skip))
        return self.conv(torch.cat([x, skip], dim=1))


# ── SegFormerUNet ──────────────────────────────────────────────────────────────

class SegFormerUNet(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        variant: str = "b2",
        pretrained: bool = True,
        use_fusion_neck: bool = False,
        use_aspp: bool = True,
        deep_supervision: bool = True,
        v1_compat: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import SegformerConfig, SegformerModel as _HFEncoder
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc

        model_id = f"nvidia/mit-{variant}"

        if pretrained:
            try:
                self.encoder = _HFEncoder.from_pretrained(model_id)
            except Exception as exc:
                print(f"[SegFormerUNet] 사전학습 로드 실패 → 랜덤 초기화: {exc}")
                config = SegformerConfig.from_pretrained(model_id)
                self.encoder = _HFEncoder(config)
        else:
            config = SegformerConfig.from_pretrained(model_id)
            self.encoder = _HFEncoder(config)

        c1, c2, c3, c4 = _ENCODER_CHANNELS[variant]
        d3, d2, d1     = _DECODER_CHANNELS[variant]

        self.use_fusion_neck  = use_fusion_neck
        self.use_aspp         = use_aspp and not use_fusion_neck
        self.deep_supervision = deep_supervision

        if use_fusion_neck:
            self.neck = FusionNeck([c1, c2, c3, c4], embed_dim=d3)
            bridge_ch = d3
            self.h4_skip_proj = nn.Sequential(
                nn.Conv2d(c1, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            )
        elif self.use_aspp:
            self.aspp = ASPPBridge(c4, d3)
            bridge_ch = d3
        else:
            bridge_ch = c4

        if v1_compat:
            self.up3 = _UpBlock(bridge_ch, c3, d3, align_channels=c3, spatial_kernel_size=3)
            self.up2 = _UpBlock(d3,        c2, d2, align_channels=c2, spatial_kernel_size=3)
            self.up1 = _UpBlock(d2,        c1, d1, align_channels=c1, spatial_kernel_size=7)
        else:
            self.up3 = _UpBlock(bridge_ch, c3, d3, spatial_kernel_size=3)
            self.up2 = _UpBlock(d3,        c2, d2, spatial_kernel_size=3)
            self.up1 = _UpBlock(d2,        c1, d1, spatial_kernel_size=7)

        self.pixel_shuffle_head = PixelShuffleHead(d1, out_channels)

        if deep_supervision:
            self.aux_head3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head2 = nn.Conv2d(d2, out_channels, kernel_size=1)

        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        h, w = x.shape[-2:]

        enc = self.encoder(pixel_values=x, output_hidden_states=True)
        h4, h8, h16, h32 = enc.hidden_states

        if self.use_fusion_neck:
            bridge = self.neck([h4, h8, h16, h32])
            h4_for_skip = self.h4_skip_proj(h4)
        elif self.use_aspp:
            bridge = self.aspp(h32)
            h4_for_skip = h4
        else:
            bridge = h32
            h4_for_skip = h4

        d3 = self.up3(bridge, h16)
        d2 = self.up2(d3,    h8)
        d1 = self.up1(d2,    h4_for_skip)

        main_logit = self.pixel_shuffle_head(d1)
        if main_logit.shape[-2:] != (h, w):
            main_logit = F.interpolate(main_logit, size=(h, w), mode="bilinear", align_corners=False)

        if self.training and self.deep_supervision:
            return {"main": main_logit, "aux3": self.aux_head3(d3), "aux2": self.aux_head2(d2)}

        return main_logit


# ── Loading & inference ───────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> SegFormerUNet:
    model = SegFormerUNet(
        variant="b2",
        pretrained=False,
        use_aspp=True,
        use_fusion_neck=False,
        deep_supervision=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def predict_mask(
    model: SegFormerUNet,
    image_pil: Image.Image,
    device: torch.device,
    image_size: int = 768,
    threshold: float = 0.5,
    min_component_area: int = 20,
    opening_iterations: int = 0,
    closing_iterations: int = 0,
) -> tuple[Image.Image, Image.Image]:
    orig_size = image_pil.size  # (W, H)

    resized = image_pil.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit)
        binary = (prob > threshold).squeeze().cpu().numpy()

    from src.marathon_route_extraction.unet import apply_postprocess
    binary = apply_postprocess(binary, opening_iterations, closing_iterations, min_component_area)

    mask_arr = (binary * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_arr, mode="L").resize(orig_size, Image.Resampling.NEAREST)
    return image_pil, mask_pil
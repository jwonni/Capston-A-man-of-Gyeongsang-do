"""
통합 예측 (추론) 스크립트
학습된 모델을 사용하여 마라톤 경로 예측
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
import yaml

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Avoid conflicts with third-party packages named "src" when running as a script.
    sys.modules.pop("src", None)

from src.core import VALID_EXTENSIONS
from src.data import apply_model_preprocess
from src.models import get_model


def build_arg_parser() -> argparse.ArgumentParser:
    """
    커맨드 라인 인자 설정
    """
    parser = argparse.ArgumentParser(description="U-Net prediction for marathon path segmentation")

    # config 파일 경로. 이 파일의 값이 기본값으로 먼저 적용되고,
    # 동일 옵션을 CLI로 다시 주면 CLI 값이 최종적으로 우선 적용됨.
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    parser.add_argument("--model-path", type=str, default="outputs/unet_trained/model_final.pt", help="Path to trained model checkpoint")
    parser.add_argument("--image-dir", type=str, default="data/test/images", help="Directory containing images to predict")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory for predictions")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")

    # 이진 마스크 생성 시 사용할 확률 threshold : 모델이 출력하는 확률값이 이 threshold보다 크면 해당 픽셀을 1(경로)로, 그렇지 않으면 0(비경로)으로 분류
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")  
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels in U-Net")
    
    return parser


def load_config_file(config_path: str) -> dict:
    """
    YAML config 파일을 읽어 dict로 반환한다.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config file must be a YAML mapping (key-value pairs).")

    return loaded


def parse_args_with_config() -> argparse.Namespace:
    """
    1) --config 위치를 먼저 파악하고
    2) config 값을 parser 기본값으로 주입한 뒤
    3) 전체 인자를 다시 파싱한다.
    """
    # 1차 파싱: config 경로만 먼저 읽기
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_arg_parser()

    # 2차 파싱 전: config 값을 기본값으로 주입
    if pre_args.config is not None:
        config_values = load_config_file(pre_args.config)
        parser.set_defaults(**config_values)

    # 3차 파싱: 최종 파싱 (CLI가 config 기본값을 덮어씀)
    return parser.parse_args()


def load_model(model_path: str, device: torch.device, base_channels: int = 32) -> Tuple[torch.nn.Module, dict]:
    """
    저장된 모델을 로드하는 함수
    
    Args:
        model_path: 모델 체크포인트 경로
        device: 모델을 로드할 디바이스
        base_channels: U-Net의 기본 채널 수
    
    Returns:
        (모델, 설정값) 튜플
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    ckpt_args = checkpoint.get("args", {})
    model_name = ckpt_args.get("model_name", "unet")

    model = get_model(
        model_name=model_name,
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, ckpt_args


def preprocess_image(image_path: Path, image_size: int, model_name: str) -> torch.Tensor:
    """
    이미지를 전처리하는 함수
    
    Args:
        image_path: 이미지 파일 경로
        image_size: 목표 이미지 크기
    
    Returns:
        전처리된 이미지 텐서 (1, 3, H, W)
    """
    # Pilow(PIL) 라이브러리로 Image를 열고, 이미지를 무조건 RGB 컬러 이미지로 통일
    # 그리고 이미지를 지정된 크기로 조절, BILINEAR(쌍선형 보간법)을 사용하여 픽셀 사이를 부드럽게 채움
    image = Image.open(image_path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    
    # PIL 이미지를 numpy 배열로 변환
    # 255.0으로 나눠줌으로써 0 ~ 255 사이 픽셀값을 0.0 ~ 1.0 사이로 정규화
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    
    # numpy 배열을 Tensor로 변환하고, (H, W, C) -> (C, H, W) 순서로 변경
    image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
    image_tensor = apply_model_preprocess(image_tensor, model_name)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict_single(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    단일 이미지에 대해 예측을 수행하는 함수
    
    Args:
        model: 학습된 모델
        image_tensor: 전처리된 이미지 텐서
        device: 계산을 수행할 디바이스
        threshold: 이진 마스크 생성 시 확률 threshold
    
    Returns:
        예측된 마스크 (H, W) numpy 배열, 값은 0 또는 255
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float().squeeze().cpu().numpy()
    
    # 0.0 또는 1.0을 0 또는 255로 변환하여 이미지로 저장 가능하게 함
    mask = (mask * 255).astype(np.uint8)
    
    return mask


def main() -> None:
    args = parse_args_with_config()

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, device, base_channels=args.base_channels)
    model_name = model_args.get("model_name", "unet")
    print(f"Model: {model_name}")

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 디렉토리에서 이미지 파일들을 찾기
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    
    if not image_paths:
        raise RuntimeError(f"No valid images found in {image_dir}")

    print(f"Found {len(image_paths)} images to predict")
    print("-" * 80)

    # 예측 수행
    predictions = []
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"Predicting {idx}/{len(image_paths)}: {image_path.name}")
        with Image.open(image_path) as original_image:
            original_size = original_image.size
        
        # 이미지 전처리
        image_tensor = preprocess_image(image_path, args.image_size, model_name=model_name)
        
        # 예측
        mask = predict_single(model, image_tensor, device, threshold=args.threshold)
        
        # 마스크를 원본 이미지 해상도로 복원한 뒤 저장
        output_name = image_path.stem + "_mask.png"
        output_path = output_dir / output_name
        
        # PIL 이미지로 변환하여 저장 - NEAREST(최근접 이웃 보간법)을 사용하여 픽셀 값을 그대로 유지하면서 크기 조절
        mask_image = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
        mask_image.save(output_path)
        
        predictions.append({
            "image": image_path.name,
            "mask": output_name,
            "output_path": str(output_path),
            "original_size": [original_size[0], original_size[1]],
        })
        
        print(f"  Mask saved to {output_path}")

    print("-" * 80)
    print(f"Prediction completed! Generated {len(predictions)} masks")

    # 예측 결과 기록 저장
    result_log_path = output_dir / "prediction_log.json"
    with open(result_log_path, "w") as f:
        json.dump({
            "total_predictions": len(predictions),
            "model_path": str(args.model_path),
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "image_size": args.image_size,
            "threshold": args.threshold,
            "predictions": predictions,
        }, f, indent=4)
    print(f"Result log saved to {result_log_path}")


if __name__ == "__main__":
    main()

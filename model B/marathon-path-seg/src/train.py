"""
통합 학습 스크립트
마라톤 경로 분할 모델 학습
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Avoid conflicts with third-party packages named "src" when running as a script.
    sys.modules.pop("src", None)



# src/__init__.py에서 UNet, MarathonSegDataset, collect_pairs, split_pairs, set_seed, run_epoch, EpochStats를 가져옴
from src.core import collect_pairs, split_pairs, set_seed, run_epoch, EpochStats
from src.data import MarathonSegDataset
from src.models import get_model
# src/losses.py에서 BCEIoULoss, BCEDiceLoss를 가져옴
from src.losses import BCEIoULoss, BCEDiceLoss



def build_arg_parser() -> argparse.ArgumentParser:
    """
    커맨드 라인 인자 설정
    """
    parser = argparse.ArgumentParser(description="U-Net training for marathon path segmentation")

    # config 파일 경로. 이 파일의 값이 기본값으로 먼저 적용되고,
    # 동일 옵션을 CLI로 다시 주면 CLI 값이 최종적으로 우선 적용됨.
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # 데이터 관련 인자
    parser.add_argument("--data-root", type=str, default="data/train", help="Training data root directory")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use-augmentation", action="store_true", help="Apply basic train-time data augmentation")
    
    # 모델 관련 인자
    parser.add_argument("--model-name", type=str, default="unet", choices=["unet", "resunet", "deeplabv3", "segformer"], help="Model architecture name")
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels in U-Net")
    
    # 학습 관련 인자
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    
    # 손실 함수 관련 인자
    parser.add_argument("--loss-type", type=str, default="bce_iou", choices=["bce_iou", "bce_dice"],
                       help="Loss function type")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE loss")
    parser.add_argument("--iou-weight", type=float, default=0.5, help="Weight for IoU loss (when using bce_iou)")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Weight for Dice loss (when using bce_dice)")
    parser.add_argument("--pos-weight", type=float, default=10.0, help="Positive weight for class imbalance")
    
    # 출력 관련 인자
    parser.add_argument("--output-dir", type=str, default="outputs/unet_trained", help="Output directory")
    parser.add_argument("--save-interval", type=int, default=1, help="Save model every N epochs")
    
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


def build_loss_fn(args: argparse.Namespace) -> torch.nn.Module:
    """
    인자에 따라 손실 함수를 생성하는 함수
    """
    if args.loss_type == "bce_iou":
        return BCEIoULoss(
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
            pos_weight=args.pos_weight,
        )
    elif args.loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            pos_weight=args.pos_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def main() -> None:
    # 터미널에서 입력받은 하이퍼파라미터(학습률, 배치사이즈 등)나 파일 경로 등의 설정값을 불러옴 
    args = parse_args_with_config()

    # 매번 같은 결과를 얻기 위해 시드 값을 고정함 - 랜덤 시드 고정은 모델의 초기 가중치, 데이터 섞는 순서 등에서 일관된 결과를 얻도록 도와줌
    set_seed(args.seed)

    # 학습에 사용할 images와 경로 이미지 masks, 그리고 학습된 모델이 저장될 출력 폴더의 경로를 설정
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디렉토리 존재 확인
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected folders not found: {images_dir} and {masks_dir}")

    print(f"Loading data from {data_root}")

    # 마라톤 이미지와 마스킹 이미지 짝짓기
    pairs = collect_pairs(images_dir, masks_dir)
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 image-mask pairs to split train/val")

    # Train과 Validation 데이터 분할
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Total pairs: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # 데이터셋 생성
    train_ds = MarathonSegDataset(
        train_pairs,
        image_size=args.image_size,
        model_name=args.model_name,
        use_augmentation=args.use_augmentation,
    )
    val_ds = MarathonSegDataset(val_pairs, image_size=args.image_size, model_name=args.model_name)

    # DataLoader 생성 - DataLoader는 데이터셋에서 배치 단위로 데이터를 불러오는 역할을 함. 학습 중에 데이터를 섞거나 여러 프로세스를 사용하여 데이터를 불러올 수 있도록 도와줌
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        print(f"Using device: {device} ({gpu_name})")
    else:
        gpu_name = "cpu"
        print(f"Using device: {device}")

    # 모델 초기화
    model = get_model(
        model_name=args.model_name,
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
    )
    model = model.to(device)

    # 손실 함수 설정
    criterion = build_loss_fn(args) # 인자에 따라 손실 함수를 생성하는 함수
    criterion = criterion.to(device) # 손실 함수를 디바이스로 이동 - 모델과 손실 함수를 같은 디바이스에 올려야 계산이 가능함 : GPU에서 모델을 학습할 때 손실 함수도 GPU로 이동시켜야 함
    print(f"Loss function: {args.loss_type}")
    print(f"Model: {args.model_name}")

    # 최적화 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습 기록 저장
    log_history = []

    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 80)

    # Training loop
    for epoch in range(args.epochs):
        # Training
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        
        # Validation
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)

        # 통계 기록
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_stats.loss,
            "train_dice": train_stats.dice,
            "train_iou": train_stats.iou,
            "val_loss": val_stats.loss,
            "val_dice": val_stats.dice,
            "val_iou": val_stats.iou,
            "device": str(device),
            "gpu_name": gpu_name,
        }
        log_history.append(log_entry)

        # 출력
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.6f} | Train Dice: {train_stats.dice:.4f} | Train IoU: {train_stats.iou:.4f} | "
            f"Val Loss: {val_stats.loss:.6f} | Val Dice: {val_stats.dice:.4f} | Val IoU: {val_stats.iou:.4f}"
        )

        # 모델 저장 (매 save_interval epoch마다)
        if (epoch + 1) % args.save_interval == 0:
            model_path = output_dir / f"model_epoch_{epoch + 1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, model_path)
            print(f"  Model saved to {model_path}")

    print("-" * 80)
    print("Training completed!")

    # 최종 모델 저장
    final_model_path = output_dir / "model_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # 학습 기록 저장
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=4)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()

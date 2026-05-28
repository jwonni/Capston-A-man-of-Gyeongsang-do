"""
Central configuration for marathon route extraction defaults.

Edit values in this file to change default behavior across the project
(`app.py` and `src.marathon_route_extraction.postprocess`).

# Notes:
# - These defaults are intended for 512x512 model output. The API
#   scales some of these (area/length) based on actual mask size.
"""

AREA_THRESH = 1000          # 이 픽셀 수 미만이면 노이즈 후보
CIRC_THRESH = 0.5           # circularity 초과 시 원형으로 간주 (0~1)
SKEL_THRESH = 400            # skeleton 길이(px) 미만이면 노이즈 후보
MAX_DISTANCE = 150.0        # fragment 연결 허용 최대 거리 (px)
MIN_FRAGMENT_SIZE = 10      # 연결 대상 fragment 최소 픽셀 수
LINE_THICKNESS = 2          # 연결선 두께 (px)
MORPH_CLOSE_SIZE = 0        # morphology closing 커널 크기. 미지정 시 입력 mask 단변의 1%%(최소 5)로 자동 계산
FINAL_SIZE_THRESH = 0       # Step 3 후 남은 fragment 중 이 픽셀 수 미만을 제거. 0=주경로만 보존.
SPUR_LENGTH = 20            # 이 픽셀 수 미만인 가지를 잔가지로 간주해 제거. 0=제거 안 함.
SKEL_MORPH_CLOSE = 0        # 스켈레톤화 전 morphology closing 커널 크기 (0=비활성화, 권장 3~7).
MIN_DIST = 8.0              # 등간격 거리 샘플링 최소 간격 (px). 클수록 더 적은 점 유지.


import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ── 세그멘테이션 모델 ────────────────────────────────────────────────────────
    # "segformer_unet_b2" 또는 "unet" — 환경변수 MODEL_TYPE으로 변경 가능
    MODEL_TYPE = os.environ.get("MODEL_TYPE", "segformer_unet_b2")
    MODEL_PATH = BASE_DIR / "weights" / (
        "segformer_unet_b2_best.pt" if MODEL_TYPE == "segformer_unet_b2" else "unet_best.pt"
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 768 if MODEL_TYPE == "segformer_unet_b2" else 512
    THRESHOLD = 0.5

    OPENING_ITERATIONS = 2
    CLOSING_ITERATIONS = 1
    MIN_COMPONENT_AREA = 1200

    # ── Hi-SAM 텍스트 분할 ───────────────────────────────────────────────────────
    HISAM_REPO_DIR = BASE_DIR / "Hi-SAM"
    HISAM_CHECKPOINT = BASE_DIR / "Hi-SAM" / "pretrained_checkpoint" / "hi_sam_l.pth"
    HISAM_MODEL_TYPE = "vit_l"          # vit_t / vit_s / vit_b / vit_l / vit_h

    HISAM_TOTAL_POINTS  = 600
    HISAM_BATCH_POINTS  = 64
    HISAM_SCORE_THRESH  = 0.4
    HISAM_PRE_NMS_TOP_K = 500
    HISAM_NMS_THRESH    = 0.6
    HISAM_MIN_POLYGON_AREA = 32

    # 크롭 마진 (픽셀): polygon bbox 확장 시 상하좌우 여백
    HISAM_CROP_X_MARGIN = 10
    HISAM_CROP_Y_MARGIN = 0

    # ── PaddleOCR 입력 전처리 ────────────────────────────────────────────────────
    PADDLE_DEVICE = "cpu"               # torch(Hi-SAM)와 CUDA 충돌 방지
    PREP_TARGET_LONG_SIDE  = 256
    PREP_RETRY_LONG_SIDE   = 512
    PREP_MIN_HEIGHT        = 64
    PREP_MIN_WIDTH         = 64
    PREP_MAX_LONG_SIDE     = 1280
    PREP_PADDING           = 16
    PREP_RETRY_PADDING     = 28

    # ── PaddleOCR ────────────────────────────────────────────────────────────────
    OCR_LANG        = "korean"
    MIN_CONFIDENCE  = 0.7
    DROP_EMPTY_TEXT = True
    SORT_BY_COORDS  = True

    # OCR 텍스트 세로 병합 파라미터
    OCR_X_MERGE_TOL = 3
    OCR_Y_MERGE_TOL = 30

    # ── 카카오 Local API ─────────────────────────────────────────────────────────
    KAKAO_API_KEY      = os.environ.get("KAKAO_API_KEY", "")
    KAKAO_MAX_WORKERS  = 10
    KAKAO_MAX_RETRIES  = 4
    KAKAO_BACKOFF_BASE = 0.7
    KAKAO_MAD_THRESH   = 2.0

    # ── 앵커 후보 필터 ───────────────────────────────────────────────────────────
    ANCHOR_MIN_TEXT_LEN    = 3
    ANCHOR_TEXT_BLACKLIST  = [
        "km", "m",
        "start", "finish",
        "출발점", "출발",
        "반환점", "반환",
        "도착점", "도착",
        "코스",
    ]

    # ── 호모그래피 재투영 오차 기반 이상치 제거 ──────────────────────────────────
    HOMOGRAPHY_MAX_ERROR_PX        = 7.0  # 2차 정제 (오프셋 보정 후)
    HOMOGRAPHY_MIN_ANCHORS         = 6

    # ── 마커 오프셋 캘리브레이션 ──────────────────────────────────────────────────
    CALIB_CONSISTENCY_PX = 3.0
    CALIB_MIN_SAMPLES    = 2

    # ── 웹 서버 ─────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB = 32
    HOST  = "0.0.0.0"
    PORT  = 8010
    DEBUG = False
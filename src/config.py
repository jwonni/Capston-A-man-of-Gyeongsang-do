import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ── 세그멘테이션 모델 ────────────────────────────────────────────────────────
    MODEL_PATH = BASE_DIR / "weights" / "model_best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 512
    THRESHOLD = 0.5

    OPENING_ITERATIONS = 2
    CLOSING_ITERATIONS = 1
    MIN_COMPONENT_AREA = 1200

    # ── Hi-SAM 텍스트 분할 ───────────────────────────────────────────────────────
    HISAM_REPO_DIR = BASE_DIR / "Hi-SAM"
    HISAM_CHECKPOINT = BASE_DIR / "Hi-SAM" / "pretrained_checkpoint" / "hi_sam_l.pth"
    HISAM_MODEL_TYPE = "vit_l"          # vit_t / vit_s / vit_b / vit_l / vit_h

    HISAM_TOTAL_POINTS = 3000
    HISAM_BATCH_POINTS = 64
    HISAM_SCORE_THRESH = 0.4
    HISAM_NMS_THRESH   = 0.6
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
    KAKAO_MAD_THRESH   = 3.0

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
    HOMOGRAPHY_MAX_ERROR_PX = 10.0
    HOMOGRAPHY_MIN_ANCHORS  = 6

    # ── 마커 오프셋 캘리브레이션 ──────────────────────────────────────────────────
    CALIB_CONSISTENCY_PX = 3.0
    CALIB_MIN_SAMPLES    = 2

    # ── 웹 서버 ─────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB = 32
    HOST  = "0.0.0.0"
    PORT  = 8010
    DEBUG = False

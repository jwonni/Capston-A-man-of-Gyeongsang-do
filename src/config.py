from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ── 세그멘테이션 모델 ────────────────────────────────────────────────────────
    MODEL_PATH = BASE_DIR / "weights" / "model_best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 입력 해상도 및 마스크 이진화 임계값
    IMAGE_SIZE = 512
    THRESHOLD = 0.5

    # 형태학 후처리 (Opening: 잔 노이즈 제거, Closing: 끊긴 선 연결)
    OPENING_ITERATIONS = 2
    CLOSING_ITERATIONS = 1
    MIN_COMPONENT_AREA = 1200

    # ── Hi-SAM 텍스트 분할 ───────────────────────────────────────────────────────
    # Hi-SAM 저장소 루트 (git clone 경로)
    # Point to the cloned Hi-SAM repo root (contains the `hi_sam` package).
    HISAM_REPO_DIR = BASE_DIR / "Hi-SAM"
    # 체크포인트 파일 경로
    HISAM_CHECKPOINT = BASE_DIR / "Hi-SAM" / "pretrained_checkpoint" / "hi_sam_l.pth"
    HISAM_MODEL_TYPE = "vit_l"          # vit_t / vit_s / vit_b / vit_l / vit_h

    # AutoMaskGenerator 파라미터 (노트북 셀 4 CONFIG와 동일)
    HISAM_TOTAL_POINTS = 600            # 전체 포그라운드 샘플링 포인트 수
    HISAM_BATCH_POINTS = 200            # 한 번에 처리할 포인트 배치 크기
    HISAM_SCORE_THRESH = 0.4            # 마스크 confidence 임계값
    HISAM_NMS_THRESH   = 0.5            # 마스크 간 NMS IoU 임계값

    # 크롭 마진 (픽셀): 텍스트 bbox 확장 시 상하좌우 여백
    HISAM_CROP_X_MARGIN = 30
    HISAM_CROP_Y_MARGIN = 8
    # 크롭 업스케일 배수 (너무 작은 크롭 이미지를 Qwen에 넘기기 전 확대)
    HISAM_UPSCALE = 2

    # ── Qwen2.5-VL OCR ──────────────────────────────────────────────────────────
    QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_MAX_NEW_TOKENS_CROP = 32
    QWEN_MAX_NEW_TOKENS_CATEGORIZE = 256
    QWEN_DROP_EMPTY = True

    # ── 카카오 Local API ─────────────────────────────────────────────────────────
    KAKAO_API_KEY = "4d43c604d0b8b2d23d960ca86a486e20"
    KAKAO_MAD_THRESH = 3.0
    KAKAO_BBOX_MARGIN = 0.01

    # ── 앵커 후보 규칙 기반 필터 ─────────────────────────────────────────────────
    ANCHOR_BLACKLIST = [
        "아파트", "오피스텔", "빌딩", "사옥", "타워", "상가",
        "고등학교", "초등학교", "중학교",
        "스",
    ]
    ANCHOR_WHITELIST = [
        "역", "국회", "공원", "병원", "학교", "박물관",
    ]
    ANCHOR_MIN_TEXT_LEN = 3

    # ── 호모그래피 재투영 오차 기반 이상치 제거 ──────────────────────────────────
    HOMOGRAPHY_MAX_ERROR_PX = 10.0
    HOMOGRAPHY_MIN_ANCHORS = 6

    # ── 웹 서버 ─────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB = 32
    HOST = "0.0.0.0"
    PORT = 8010
    DEBUG = False

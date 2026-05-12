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
    # Hugging Face 모델 ID (Qwen2.5-VL 7B Instruct)
    QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    # 크롭 OCR 최대 생성 토큰 (단어 수준이므로 짧게)
    QWEN_MAX_NEW_TOKENS_CROP = 32
    # 카테고리 분류 최대 생성 토큰 (JSON 출력용)
    QWEN_MAX_NEW_TOKENS_CATEGORIZE = 256
    # 빈 결과(텍스트 없는 크롭) 제외 여부
    QWEN_DROP_EMPTY = True

    # ── 카카오 Local API ─────────────────────────────────────────────────────────
    # 카카오 REST API 앱 키 (https://developers.kakao.com)
    KAKAO_API_KEY = "4d43c604d0b8b2d23d960ca86a486e20"                  # 배포 시 환경변수 KAKAO_API_KEY로 주입
    # MAD 이상치 제거 임계값 (중앙값에서 MAD의 몇 배 이상 떨어지면 이상치)
    KAKAO_MAD_THRESH = 3.0
    # 동적 bbox 재검색 시 인라이어 경계에 추가할 경위도 여백
    KAKAO_BBOX_MARGIN = 0.01

    # ── 앵커 후보 규칙 기반 필터 ─────────────────────────────────────────────────
    # 앵커로 사용하기 너무 모호한 단어 목록 (정확 매칭)
    ANCHOR_BLACKLIST = [
        "아파트", "오피스텔", "빌딩", "사옥", "타워", "상가",
        "고등학교", "초등학교", "중학교",
        "스",                                    # OCR 오류 단편
    ]
    # 이 키워드가 포함된 텍스트는 앵커 우선 채택 (부분 일치)
    ANCHOR_WHITELIST = [
        "역",                                    # 지하철역
        "국회", "공원", "병원", "학교", "박물관",
        "IFC", "KBS", "KRX", "IBK",
        "콘래드", "파크원", "페어몬트", "켄싱턴", "CCMM",
    ]
    # 앵커 텍스트 최소 길이 (이 미만이면 제외)
    ANCHOR_MIN_TEXT_LEN = 3

    # ── 호모그래피 재투영 오차 기반 이상치 제거 ──────────────────────────────────
    # 재투영 오차(픽셀)가 이 값을 초과하면 해당 앵커 제거
    HOMOGRAPHY_MAX_ERROR_PX = 10.0
    # 앵커가 이 수 이하로 줄어들면 강제 중단 (호모그래피 최소 요구 4개)
    HOMOGRAPHY_MIN_ANCHORS = 6

    # ── 웹 서버 ─────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB = 32
    HOST = "0.0.0.0"
    PORT = 8010
    DEBUG = False

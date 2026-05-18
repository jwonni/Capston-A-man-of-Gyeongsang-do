# Marathon Route Extraction Web App

FastAPI 백엔드 + 단일 페이지 HTML 프론트엔드로 마라톤 코스 이미지에서 경로를 추출하고, 지리좌표(GPX)로 변환하는 캡스톤 프로젝트입니다.

## 주요 기능

1. **경로 마스크 추출** — U-Net 모델로 마라톤 코스 이미지에서 경로 마스크 예측
2. **후처리** — 연결 성분 필터링 + 스켈레톤화
3. **경로 추출** — 스켈레톤에서 시작점→끝점 순서 경로 추출
4. **지리좌표 변환** — Hi-SAM 텍스트 감지 → PaddleOCR → 카카오 Local API → 호모그래피로 픽셀↔GPS 변환
5. **GPX 출력** — 변환된 GPS 좌표를 GPX 파일로 내보내기

## 프로젝트 구조

```
├── app.py                              # FastAPI 서버 진입점
├── requirements.txt                    # Python 의존 패키지
├── .env                                # 환경변수 (KAKAO_API_KEY 등, git 제외)
├── .env.example                        # 환경변수 예시
├── Hi-SAM/                             # Hi-SAM 텍스트 감지 라이브러리 (별도 클론 필요)
│   └── pretrained_checkpoint/
│       ├── hi_sam_l.pth                # Hi-SAM 체크포인트 (별도 다운로드 필요)
│       └── sam_vit_l_0b3195.pth        # SAM 기본 체크포인트 (별도 다운로드 필요)
├── docs/
│   ├── HISAM_CHECKPOINT_DOWNLOAD.md    # Hi-SAM 체크포인트 다운로드 가이드
│   └── SETUP_GEOREF.md                 # 지리좌표 변환 환경 설정 가이드
├── scripts/
│   └── visualize_component_filter.py   # 연결 성분 필터 디버그 시각화
├── src/
│   ├── config.py                       # 전역 설정 (Config 클래스)
│   ├── georeferencing/
│   │   ├── anchor_builder.py           # 카카오 API 앵커 구축 + MAD 필터
│   │   ├── homography.py               # 픽셀↔GPS 호모그래피 계산
│   │   ├── ocr.py                      # PaddleOCR 기반 텍스트 인식 파이프라인
│   │   └── text_detector.py            # Hi-SAM 텍스트 영역 감지
│   ├── gpx_conversion/
│   │   └── gpx_converter.py            # 픽셀 경로 → GPX 변환
│   └── marathon_route_extraction/
│       ├── component_filter.py         # 연결 성분 필터링
│       ├── model.py                    # U-Net 모델 / 추론
│       ├── path_extractor.py           # 스켈레톤 → 순서 있는 경로 추출
│       └── postprocess.py              # 4단계 후처리 파이프라인
├── static/
│   └── index.html                      # 5단계 데모 UI
└── weights/
    └── model_best.pt                   # 학습된 U-Net 가중치
```

## 설치 및 실행

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Hi-SAM 설정 (지리좌표 변환 사용 시)

```bash
# 저장소 클론
git clone https://github.com/ymy-k/Hi-SAM.git Hi-SAM

# Hi-SAM 의존 패키지 설치 (Hi-SAM/requirements.txt 대신 사용 — Python 3.13 호환)
pip install einops timm pycocotools absl-py python-Levenshtein tqdm

# 체크포인트 다운로드
curl -L -o Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
curl -L -o Hi-SAM/pretrained_checkpoint/hi_sam_l.pth https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_l.pth
```

### 3. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에 카카오 REST API 키 입력
```

```env
KAKAO_API_KEY=여기에_카카오_API_키_입력
```

### 4. 서버 실행

```bash
python app.py
```

### 5. 브라우저 접속

```
http://localhost:8010
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/predict` | 이미지 업로드 → 경로 마스크 예측 |
| `POST` | `/api/postprocess` | 연결 성분 필터링 + 스켈레톤화 |
| `POST` | `/api/extract_path` | 시작점→끝점 순서 경로 추출 |
| `POST` | `/api/georeference` | Hi-SAM + PaddleOCR + 카카오 API → 호모그래피 계산 |
| `POST` | `/api/convert_gpx` | 픽셀 경로 → GPX 파일 변환 |
| `GET`  | `/api/health` | 서버 상태 확인 |

## 지리좌표 변환 파이프라인 (`/api/georeference`)

```
이미지
  ↓ Hi-SAM (vit_l)
텍스트 영역 polygon
  ↓ PaddleOCR (korean, CPU)
인식된 텍스트 + 좌표
  ↓ 카카오 Local API (병렬 검색 + MAD 필터)
앵커 (픽셀 좌표 ↔ GPS 좌표) 쌍
  ↓ 호모그래피 + 재투영 오차 기반 이상치 제거
픽셀↔GPS 변환 행렬
```

## 유틸리티

연결 성분 필터 결과 시각화:

```bash
python scripts/visualize_component_filter.py --input-mask sample_route.png --min-component-area 130
```

## 참고 사항

- GPU가 없으면 CPU로 자동 전환됨 (Hi-SAM 추론은 시간이 오래 걸릴 수 있음)
- PaddleOCR은 CPU 전용으로 고정 (Hi-SAM의 torch GPU와 CUDA 충돌 방지)
- Windows 환경에서는 PaddlePaddle `3.0 이상 3.3 미만` 버전 사용 권장 (oneDNN 버그 회피)

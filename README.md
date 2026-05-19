# Marathon Route Extraction Web App

FastAPI 백엔드 + 단일 페이지 HTML 프론트엔드로 마라톤 코스 이미지에서 경로를 추출하고, 지리좌표(GPX)로 변환하는 캡스톤 프로젝트입니다.

## 주요 기능

1. **경로 마스크 추출** — U-Net 모델로 마라톤 코스 이미지에서 경로 마스크 예측
2. **후처리** — 연결 성분 필터링 + 스켈레톤화
3. **그래프 단순화** — 스켈레톤을 5단계 파이프라인으로 의미론적 그래프 G′로 압축
4. **경로 추출** — G′에서 BFS 탐색 후 엣지 픽셀 복원 + 꺾임점 기반 키포인트 추출
5. **지리좌표 변환** — Hi-SAM 텍스트 감지 → PaddleOCR → 카카오 Local API → 호모그래피로 픽셀↔GPS 변환
6. **GPX 출력** — 변환된 GPS 좌표를 GPX 파일로 내보내기

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
│       ├── graph_simplifier.py         # 5단계 그래프 단순화 파이프라인
│       ├── path_extractor.py           # 스켈레톤 → 순서 있는 경로 추출
│       ├── postprocess.py              # 4단계 후처리 파이프라인
│       ├── segformer_unet_b2.py        # SegFormer-UNet B2 모델 / 추론
│       └── unet.py                     # U-Net 모델 / 추론
├── static/
│   └── index.html                      # 5단계 데모 UI
└── weights/
    ├── segformer_unet_b2_best.pt       # SegFormer-UNet B2 가중치
    └── unet_best.pt                    # U-Net 가중치
```

## 설치 및 실행

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 경로 추출 모델 가중치 다운로드 (SegFormer-UNet B2)

U-Net과 SegFormer-UNet-B2 가중치는 아래 Google Drive에서 다운

- 다운로드 링크: https://drive.google.com/drive/folders/1TKDRnaR8HlrcM2B8wclnDiv_GN5hD2gW?usp=sharing

Drive에 저장된 파일명이 아래와 같다면:

- `model_best.pt`
- `model_last.pt`

프로젝트에서는 아래처럼 이름을 바꿔 `weights/` 폴더에 저장

- 각각 `model_best.pt` -> `segformer_unet_b2_best.pt`와 `unet_best.pt`으로 변경

### 3. Hi-SAM 설정 (지리좌표 변환 사용 시)

```bash
# 저장소 클론
git clone https://github.com/ymy-k/Hi-SAM.git Hi-SAM

# Hi-SAM 의존 패키지 설치 (Hi-SAM/requirements.txt 대신 사용 — Python 3.13 호환)
pip install einops timm pycocotools absl-py python-Levenshtein tqdm

# 체크포인트 다운로드
curl -L -o Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
curl -L -o Hi-SAM/pretrained_checkpoint/hi_sam_l.pth https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_l.pth
```

### 4. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에 카카오 REST API 키 입력
```

```env
KAKAO_API_KEY=여기에_카카오_API_키_입력
```

### 5. 서버 실행

```bash
python app.py
```

### 6. 브라우저 접속

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

## 경로 추출 파이프라인 (`/api/predict` → `/api/postprocess` → `/api/extract_path`)

```
마라톤 코스 이미지
  ↓ /api/predict
경로 추출 모델 (SegFormer-UNet B2 / UNet)
  ↓
경로 마스크
  ↓ /api/postprocess
후처리 (노이즈 제거 + fragment 연결 + 스켈레톤화)
  ↓
스켈레톤 이미지
  ↓ /api/extract_path  — graph_simplifier.py
그래프 단순화 5단계
  1. 1차 선형 압축    — degree-2 중간 노드 제거
  2. 노드 병합        — 유클리드 거리 ≤ τ 인 노드를 centroid로 통합
  3. 2차 선형 압축    — 병합 후 새로 생긴 degree-2 노드 재압축
  4. 컴포넌트 연결    — 분리 성분을 가장 가까운 노드쌍으로 반복 연결
  5. 핵심 노드 유지   — leaf(degree 1) + junction(degree ≥ 3)만 남은 G′
  ↓
단순화된 그래프 G′에서 BFS 경로 탐색 (start → end)
  ↓
엣지 픽셀 복원 + 꺾임점 추출 (각도 기반 키포인트)
  ↓
경로 픽셀 좌표 리스트 [(x, y), ...]
```

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

## 유틸리티 (디버깅)

연결 성분 필터 결과 시각화:

```bash
python scripts/visualize_component_filter.py --input-mask sample_route.png --min-component-area 130
```

## 참고 사항

- GPU가 없으면 CPU로 자동 전환됨 (Hi-SAM 추론은 시간이 오래 걸릴 수 있음)
- PaddleOCR은 CPU 전용으로 고정 (Hi-SAM의 torch GPU와 CUDA 충돌 방지)
- Windows 환경에서는 PaddlePaddle `3.0 이상 3.3 미만` 버전 사용 권장 (oneDNN 버그 회피)

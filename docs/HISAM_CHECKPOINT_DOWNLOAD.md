# Hi-SAM 체크포인트 다운로드 가이드

`/api/georeference` 엔드포인트를 사용하려면 Hi-SAM 사전 학습 모델 체크포인트가 필요합니다.

## 다운로드 링크

Hi-SAM 공식 저장소 (https://github.com/IDEA-Research/Hi-SAM) 에서 제공하는 체크포인트:

- **vit_l** (Large, ~1.1GB, 권장):
  https://huggingface.co/IDEA-Research/Hi-SAM/resolve/main/hi_sam_l.pth

- **vit_b** (Base, ~600MB):
  https://huggingface.co/IDEA-Research/Hi-SAM/resolve/main/hi_sam_b.pth

- **vit_s** (Small, ~400MB):
  https://huggingface.co/IDEA-Research/Hi-SAM/resolve/main/hi_sam_s.pth

- **vit_t** (Tiny, ~300MB):
  https://huggingface.co/IDEA-Research/Hi-SAM/resolve/main/hi_sam_t.pth

## 설치 위치

다운로드한 `.pth` 파일을 여기에 놓으세요:

```
프로젝트루트/
├── Hi-SAM/
│   ├── pretrained_checkpoint/
│   │   ├── hi_sam_l.pth          ← 여기에 배치
│   │   └── ...
```

## 방법 1: 브라우저에서 다운로드

1. 위 링크 중 원하는 모델 클릭 (기본값은 vit_l)
2. Hugging Face 페이지에서 "Download" 버튼 클릭
3. `Hi-SAM/pretrained_checkpoint/` 폴더에 저장

## 방법 2: 명령줄에서 다운로드 (PowerShell)

```powershell
# 폴더 생성
mkdir "Hi-SAM\pretrained_checkpoint"

# vit_l 다운로드 예시 (curl 또는 wget 사용)
$url = "https://huggingface.co/IDEA-Research/Hi-SAM/resolve/main/hi_sam_l.pth"
$output = "Hi-SAM\pretrained_checkpoint\hi_sam_l.pth"
Invoke-WebRequest -Uri $url -OutFile $output
```

## 설정 변경

`src/config.py`에서 모델 타입을 변경할 수 있습니다:

```python
HISAM_MODEL_TYPE = "vit_l"  # "vit_t", "vit_s", "vit_b", "vit_l", "vit_h" 중 선택
HISAM_CHECKPOINT = BASE_DIR / "Hi-SAM" / "pretrained_checkpoint" / "hi_sam_l.pth"
```

## 확인

파일이 올바르게 배치되었는지 확인:

```powershell
ls "Hi-SAM\pretrained_checkpoint\*.pth"
```

파일이 보이면 정상입니다.

## 문제 해결

- Hugging Face에서 다운로드 속도가 느린 경우: 미러 사이트 또는 VPN 사용 고려
- `hf_hub_download` 사용: Python에서 `huggingface_hub` 패키지를 설치 후 스크립트로 다운로드

```python
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    repo_id="IDEA-Research/Hi-SAM",
    filename="hi_sam_l.pth",
    local_dir="Hi-SAM/pretrained_checkpoint"
)
```

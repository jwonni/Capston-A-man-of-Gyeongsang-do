# Georeferencing (Hi-SAM / Qwen) Setup

This project uses Hi-SAM for text region detection and Qwen2.5-VL for OCR. Both require extra setup beyond pip installs.

1) Hi-SAM
 - Clone the Hi-SAM repository into the project root expected by `Config.HISAM_REPO_DIR` (default: `Hi-SAM`):

```bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit
git clone https://github.com/IDEA-Research/Hi-SAM.git Hi-SAM
# place the Hi-SAM model checkpoint at: Hi-SAM/pretrained_checkpoint/hi_sam_l.pth
```

Hi-SAM is imported as a local module (`hi_sam`) by `src/georeferencing/text_detector.py`. It is not a PyPI package.

2) Qwen (Qwen2.5-VL)
 - Install Hugging Face `transformers`, `accelerate`, and `safetensors` (already in `requirements.txt`).
 - The OCR wrapper expects `qwen_vl_utils` utilities; if available via pip you can install it, otherwise follow the Qwen provider instructions to prepare the model ID specified in `src/config.py`.

3) API Key
 - Set `KAKAO_API_KEY` environment variable for `/api/georeference` to work.

4) Notes
 - If you do not need georeferencing, you can skip Hi-SAM/Qwen; `/api/convert_gpx` will still produce a GPX file but coordinates will be raw pixel values.
